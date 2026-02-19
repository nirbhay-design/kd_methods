import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from data import get_cifar10_loaders
from model import StudentCNN, TeacherResNet
import warnings; warnings.filterwarnings("ignore")

def print_args(args):
    print("\n" + "="*50)
    print(f"{'Parameter':<25} | {'Value':<20}")
    print("="*50)
    for k, v in vars(args).items():
        print(f"{k:<25} | {str(v):<20}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Naive Knowledge Distillation')
    parser.add_argument('--teacher_arch', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--teacher_path', type=str, default="saved_model/teacher_resnet18_best.pth")
    parser.add_argument('--batch_size', type=int, default=128, help='Optimal for CIFAR10')
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD Temperature parameter')
    parser.add_argument('--alpha', type=float, default=0.9, help='Weight for KD loss')
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    print_args(args)

    trainloader, testloader = get_cifar10_loaders(args.batch_size)

    teacher = TeacherResNet(arch=args.teacher_arch).to(args.device)
    teacher.eval() # Teacher remains frozen
    # Note: In a real scenario, load your pre-trained teacher weights here
    print(teacher.load_state_dict(torch.load(args.teacher_path, map_location=args.device)))
    
    student = StudentCNN().to(args.device)

    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    best_acc = 0.0

    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            
            logits_s, _ = student(inputs)
            with torch.no_grad():
                logits_t, _ = teacher(inputs)

            # Hard Loss
            loss_ce = criterion_ce(logits_s, targets)
            
            # Soft Loss (KD)
            log_preds_s = F.log_softmax(logits_s / args.temperature, dim=1)
            preds_t = F.softmax(logits_t / args.temperature, dim=1)
            loss_kd = criterion_kd(log_preds_s, preds_t)
            
            # Combined Loss
            loss = (1. - args.alpha) * loss_ce + (args.alpha * (args.temperature ** 2)) * loss_kd
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        student.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                logits_s, _ = student(inputs)
                
                loss = criterion_ce(logits_s, targets)
                test_loss += loss.item()
                
                _, predicted = logits_s.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        test_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {running_loss/len(trainloader):.4f} | "
              f"Test Loss: {test_loss/len(testloader):.4f} | "
              f"Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc

    print(f"\nDistillation complete. Best Student Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()