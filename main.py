from utils.data_loader import load_wesad_all
from utils.dataset import StressWindowDataset

def main():
    subjects = load_wesad_all("data/raw/WESAD")  # requires at least a couple subject pkls
    ds = StressWindowDataset(subjects)
    print("Total windows:", len(ds))
    if len(ds) > 0:
        x, y = ds[0]
        print("Sample X shape:", x.shape, "y:", int(y))

if __name__ == "__main__":
    main()