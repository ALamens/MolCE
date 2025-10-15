# MolCE
Contrastive explanations for machine learning predictions in chemistry
# Exemplary usage:
from MolCE import MolCE

mol = Chem.MolFromSmiles(smile)

contr = MolCE.MolContrast(data_smiles=reference_dataset, model=model, predict_func=predict_func, predict_func_proba=predict_func_proba)

## substituent foils
pn_r_cpd =  contr.get_contrastive_rgroups(mol, foil_class=foil)

## scaffold foils
pn_core_cpd =  contr.get_contrastive_cores(mol, foil_class=foil, similarity_threshold=0.85)


# If a new conda environment is required:

## Set up `conda` environment

Run the following command to set up the "conda" environment for the project:

conda env create -n {insert_name_of_choice} --file environment.yml

for cross platform compatibility please use the no build file to get rid of dependencies

conda env create -n environment_cf --environment_no_build.yml

## Activate conda environment

conda activate {insert_name_of_choice}
