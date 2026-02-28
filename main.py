from utils.data_loader import load_wesad_all
from utils.dataset import StressWindowDataset

def main():
    print("Loading subjects S2–S5...")
    subjects = load_wesad_all("data/raw/WESAD", subjects=["S2", "S3", "S4", "S5"])
    print("Loaded:", [s.subject_id for s in subjects])

    print("Building windowed dataset...")
    ds = StressWindowDataset(subjects)

    print("Total windows:", len(ds))
    if len(ds) > 0:
        x, y = ds[0]
        print("Sample X shape:", x.shape, "| y:", int(y))

if __name__ == "__main__":
    main()