#!/usr/bin/env python3
"""
Smoke test for BusterNet stage-wise training implementation
"""

import os
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from model_busternet import build_busternet_full
from model_simidet import SimiDetBranch
from model_manidet import ManiDetBranch

def test_model_creation():
    """Test that we can create all the models"""
    print("Testing model creation...")
    
    try:
        # Test ManiDet
        manidet = ManiDetBranch()
        print("[PASS] ManiDet created successfully")
        
        # Test SimiDet
        simidet = SimiDetBranch()
        print("[PASS] SimiDet created successfully")
        
        # Test BusterNet
        busternet = build_busternet_full()
        print("[PASS] BusterNet created successfully")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error creating models: {e}")
        return False

def test_forward_pass():
    """Test forward pass through models"""
    print("\nTesting forward pass...")
    
    try:
        # Create fake data
        batch_size = 2
        fake_data = {
            'image': torch.randn(batch_size, 3, 256, 256),
            'mc': torch.randint(0, 3, (batch_size, 256, 256)).long().unsqueeze(1),
            'mm': torch.randint(0, 2, (batch_size, 1, 256, 256)).float(),
            'loss_mask': torch.ones(batch_size, 1, 256, 256).float()
        }
        
        # Test ManiDet
        manidet = ManiDetBranch()
        manidet_output = manidet(fake_data['image'])
        print(f"[PASS] ManiDet forward pass: {manidet_output['m_pred'].shape}")
        
        # Test SimiDet
        simidet = SimiDetBranch()
        simidet_output = simidet(fake_data['image'])
        print(f"[PASS] SimiDet forward pass: {simidet_output['m_pred'].shape}")
        
        # Test BusterNet
        busternet = build_busternet_full()
        busternet_output = busternet(fake_data['image'])
        print(f"[PASS] BusterNet forward pass:")
        print(f"  mc_pred: {busternet_output['mc_pred'].shape}")
        print(f"  simi_feat: {busternet_output['simi_feat'].shape}")
        print(f"  mani_feat: {busternet_output['mani_feat'].shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error in forward pass: {e}")
        return False

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    try:
        # Create fake data
        batch_size = 2
        fake_data = {
            'image': torch.randn(batch_size, 3, 256, 256),
            'mc': torch.randint(0, 3, (batch_size, 256, 256)).long().unsqueeze(1),
            'mm': torch.randint(0, 2, (batch_size, 1, 256, 256)).float(),
            'loss_mask': torch.ones(batch_size, 1, 256, 256).float()
        }
        
        # Test ManiDet training step
        manidet = ManiDetBranch()
        manidet.train()
        
        optimizer = torch.optim.Adam(manidet.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        output = manidet(fake_data['image'])
        loss = torch.nn.functional.binary_cross_entropy(
            output['m_pred'], 
            fake_data['mm'], 
            weight=fake_data['loss_mask']
        )
        loss.backward()
        optimizer.step()
        
        print(f"[PASS] ManiDet training step: loss = {loss.item():.4f}")
        
        # Test BusterNet training step (using the basic model)
        busternet = build_busternet_full()
        busternet.train()
        
        optimizer = torch.optim.Adam(busternet.parameters(), lr=0.0001)
        optimizer.zero_grad()
        
        output = busternet(fake_data['image'])
        # For the basic model, we need to compute loss from probabilities
        # Convert to log-like values for NLL loss, or use MSE with targets
        # For simplicity, let's just test that we can compute some loss
        pred_probs = output['mc_pred']  # [B, 3, H, W]
        target = fake_data['mc'].squeeze(1)  # [B, H, W] with values 0,1,2
        
        # Use negative log likelihood loss
        loss = torch.nn.functional.nll_loss(
            torch.log(pred_probs + 1e-8),  # Add small epsilon to avoid log(0)
            target,
            reduction='none'
        )
        # Apply loss mask
        loss_mask = fake_data['loss_mask'].squeeze(1)
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        loss.backward()
        optimizer.step()
        
        print(f"[PASS] BusterNet training step: loss = {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("BUSTERNET SMOKE TEST")
    print("="*50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Training Step", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
            status = "PASS" if result else "FAIL"
            print(f"  Result: {status}")
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            results.append(False)
    
    print("\n" + "="*50)
    if all(results):
        print("ALL SMOKE TESTS PASSED!")
        print("The implementation appears to be working correctly.")
    else:
        print("SOME SMOKE TESTS FAILED!")
        print("Please check the implementation.")
    print("="*50)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)