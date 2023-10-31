import yaml

with open ('../../params.yaml','r') as file:
    params = yaml.safe_load(file)

print(params)