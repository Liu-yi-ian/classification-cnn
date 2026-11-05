"""Pretrain the CNN, then fine-tune and test its twelve-class classifier."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for stage in ['pretrain', 'finetune']:
        command = commands.add_parser(stage)
        command.add_argument('--config', default=f'configs/{stage}.json')
        command.add_argument('--output', default=f'outputs/{stage}')
        command.add_argument('--device', default='cuda')
        command.add_argument('--resume', action='store_true')
        if stage == 'finetune':
            command.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    # The same training loop handles both stages; the configuration chooses
    # the dataset, loss settings, optimizer, and trainable parameter groups.
    from src.train import run, test_checkpoint
    best = run(args.config, args.output, getattr(args, 'checkpoint', None),
               args.resume, device=args.device)
    if args.command == 'finetune':
        # Test evaluation happens after validation-based model selection.
        result = test_checkpoint(best, Path(args.output) / 'test_metrics.json', args.device)
        print(f"test accuracy={result['accuracy']:.4f} "
              f"macro_recall={result['macro_recall']:.4f} "
              f"insect_accuracy={result['insect_accuracy']:.4f}")


if __name__ == '__main__':
    main()
