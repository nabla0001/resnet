import torch
from torchsummary import summary
from model import ResNet, ResidualBlock, PlainBlock, ZeroPadding, ZeroPaddingMaxPool, Conv1x1Projection
from data import cifar10_data_loaders
from utils import evaluate_error, save_experiment

from datetime import datetime
import argparse
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Re-produces CIFAR10 experiments from He et al. 2015.')
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--n',
                        type=int,
                        default=3,
                        help='number of stacked blocks, determines network depth. e.g. n=3/9/18 yields ResNet-20/56/110')
    parser.add_argument('--model-type', type=str, choices=['resnet', 'plain'], default='resnet')
    parser.add_argument('--skip-connection',
                        type=str,
                        choices=['none', 'zeropad', 'zeropad-maxpool', 'conv1x1-proj'],
                        default='none')
    parser.add_argument('--exp-dir', type=str, default='experiments')
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()
    print(args)

    # experiment & checkpoint tracking
    exp_path = Path(args.exp_dir) / args.exp_name
    exp_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_path = exp_path / (timestamp + '.pkl')
    checkpoint_path = exp_path / (timestamp + '.ckpt')

    print(f'experiment: {args.exp_name}')
    print(f'exp dir: {exp_path}')
    print(f'results path: {experiment_path}')
    print(f'checkpoint path: {checkpoint_path}')

    # device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'device: {device}')

    # hyperparameters (see He et al. 2015)
    batch_size = 128
    learning_rate = 0.01 if (args.n == 18) else 0.1
    num_epochs = 200
    # learning rate schedule:
    #   /10 @ 32k and 48k iterations
    #   stop training @ 64k iterations
    #   ResNet110: 0.01 -> 0.1 @ 400 iterations -> same as above
    momentum = 0.9
    weight_decay = 1e-4

    # data
    train_loader, val_loader, test_loader = cifar10_data_loaders(batch_size, data_dir=args.data_dir)

    # model
    block = ResidualBlock if args.model_type == 'resnet' else PlainBlock

    skip_connection = {'none': None,
                       'zeropad': ZeroPadding,
                       'zeropad-maxpool': ZeroPaddingMaxPool,
                       'conv1x1-proj': Conv1x1Projection}.get(args.skip_connection)

    model = ResNet(block, skip_connection, (args.n, args.n, args.n))

    print(summary(model, (3, 32, 32)))
    model.to(device)

    experiment = {
        'name': args.exp_name,
        'train_error': [],
        'val_error': [],
        'test_error': None,
        'batch': [],
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'momentum': momentum,
            'weight_decay': weight_decay,
        },
        'args': args
    }

    # loss & optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # for updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # train
    n_batches = len(train_loader)
    total_batch_count = 0
    curr_lr = learning_rate
    stop_training = False

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            total_batch_count += 1

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # decay learning rate

            # for ResNet110: start with 0.01, then go back to 0.1 after ca 400 batches
            if (args.n == 18) and (total_batch_count == 400):
                print(f'Iter {total_batch_count}: changing learning rate to 0.1')
                update_lr(optimizer, 0.1)

            if total_batch_count in (32e3, 48e3):
                curr_lr /= 10
                print(f'Iter {total_batch_count}: decreasing learning rate to {curr_lr}')
                update_lr(optimizer, curr_lr)

            if total_batch_count == 64e3:
                print(f'Iter {total_batch_count}: stopping training.')
                stop_training = True
                break

        # evaluate train/val error
        model.eval()

        train_error = evaluate_error(model, train_loader, device, 20)
        val_error = evaluate_error(model, val_loader, device, 20)

        print(f'Epoch [{epoch+1:04d}/{num_epochs:04d}] Iter [{total_batch_count:06d}] Train error: {100*train_error:.2f}% Val error: {100*val_error:.2f}%')

        model.train()

        experiment['train_error'].append(train_error)
        experiment['val_error'].append(val_error)
        experiment['batch'].append(total_batch_count)

        if stop_training:
            break

    # evaluate model
    test_error = evaluate_error(model, test_loader, device)
    experiment['test_error'] = test_error
    print(f'Test error: {100*test_error:.3f}%')

    # save results
    save_experiment(experiment, experiment_path)

    # Save the model checkpoint
    torch.save(model.state_dict(), checkpoint_path)