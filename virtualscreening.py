import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn
# Use a relative path assuming the CSV file is inside a "data" folder in your repo
file_path = "data/coconut_05_2025.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Show the first few rows
df.head()

# Select only the relevant columns
df_filtered = df[['canonical_smiles', 'name']]  # replace 'name' with the actual column name if different

# Preview the new DataFrame
print(df_filtered.head())

# 1. Define your model architecture (should match what you used during training)
class BRD4Model(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, mid_dim=128):
        super(BRD4Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


import torch
from your_model_definition import BRD4Model  # Replace with actual module if needed

# Use relative path assuming the model is stored in a "models" folder in your repo
model_path = "models/BRD4_model.pth"

# Load the model
model = BRD4Model()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Use map_location for portability
model.eval()

# 3. Load your dataset (SMILES and names)
data_path = r"E:\Projects\CADD\BRD4\coconut_05_2025.csv"  # replace with actual file
df = pd.read_csv(data_path)
df = df[['canonical_smiles', 'name']]  # adjust if your name column is called something else

df.head()

# 4. Function to convert SMILES to fingerprint
def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# 5. Predict pIC50 for each compound
results = []
for idx, row in df.iterrows():
    smiles = row['canonical_smiles']
    name = row['name']
    fp = smiles_to_fp(smiles)
    if fp is not None:
        x = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = model(x).item()
        results.append({'name': name, 'canonical_smiles': smiles, 'predicted_pIC50': prediction})


# 6. Convert to DataFrame
results_df = pd.DataFrame(results)

filtered.head()
# 7. Filter pIC50 > 7.5
filtered = results_df[results_df['predicted_pIC50'] > 7.5]
