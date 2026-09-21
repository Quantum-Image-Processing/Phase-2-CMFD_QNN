print("TEST WITH IMPORTS START")

print("About to import os")
import os
print("Os imported")

print("About to import torch")
import torch
print("Torch imported")

print("About to import sys and modify path")
import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')
print("Path modified")

print("About to import local modules")
from model_busternet import BusterNet, build_busternet_full
from model_simidet import build_simidet
from model_manidet import build_manidet
from preprocessing import BusterNetPreprocess
from dataset import SyntheticCMFDataset
print("Local modules imported")

print("About to define main")
def main():
    print("MAIN FUNCTION START")
    
print("About to call main")
main()
print("SCRIPT END")