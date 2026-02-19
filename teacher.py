import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from data import get_cifar10_loaders
from model import TeacherResNet, StudentCNN

def print_args(args):
    print("\n" + "="*50)
    print(f"{'Parameter':<25} | {'Value':<20}")
    print("="*50)
    for k, v in vars(args).items():
        print(f"{k:<25} | {str(v):<20}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model on CIFAR-10')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs to train')
    parser.add_argument('--save_dir', type=str, default='saved_model', help='Directory to save checkpoints')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--student', action="store_true", help="train student or not")
    args = parser.parse_args()

    print_args(args)

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    trainloader, testloader = get_cifar10_loaders(args.batch_size)

    # Initialize the teacher model defined in teacher.py
    if args.student:
        print("loading student model ...")
        model = StudentCNN().to(args.device)
        model_name = "student"
    else:
        print("loading teacher model ...")
        model_name = f"teacher_{args.arch}"
        model = TeacherResNet(arch=args.arch).to(args.device)

    # Standard setup for ResNet on CIFAR-10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        scheduler.step()
        
        # Validation Loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                
                test_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss/len(testloader):.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save checkpoint if it's the best performing model so far
        if test_acc > best_acc:
            print(f"---> Saving model... (Accuracy improved from {best_acc:.2f}% to {test_acc:.2f}%)")
            save_path = os.path.join(args.save_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            best_acc = test_acc

    # Save the final epoch weights just in case
    final_path = os.path.join(args.save_dir, f'{model_name}_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()