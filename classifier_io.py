import json
import numpy as np

class ClassifierModel:
    def __init__(self, model, sklearn_version=None):
        self.model = model
        self.model_params = model.__dict__
        self.set_sklearn_version(sklearn_version)

    def save_json(self, filepath):
        for k, v in self.model_params.items():
            if isinstance(v, np.ndarray):
                self.model_params[k] = v.tolist()
        json_text = json.dumps(self.model_params)
        with open(filepath, 'w') as file:
            file.write(json_text)

    def load_json(self, filepath):
        with open(filepath, 'r') as file:
            self.model_params = json.load(file)
        for k, v in self.model_params.items():
            if isinstance(v, list):
                self.model_params[k] = np.asarray(v)
        self.model.__dict__ = self.model_params
        return self

    def get(self):
        return self.model

    def set_sklearn_version(self, vstring):
        if vstring:
            self.model_params.update({"sklearn_version":vstring})
        else:
            raise Warning("The sklearn version is undefined. This information is critical for resolving future compatibility issues.")
        
    def define_features(self, dims, coords):
        """
        xarray-style definition of matrix dimensions and coordinates, for interpretability of classifier weights
        
        Classifier weights are stored as a 1d list representing a multidimensional feature set. first weight corresponds 
        to dim_0[0]-dim_1[0] feature pair, second weight corresponds to dim_0[0]-dim_1[1] and so on.  

        Parameters:
        dims - list (ordered) of feature names or "dimensions", like "channel" or "frequency". The first dimension
                designates the "outer loop" for iterating through the feature set.
        coords - dictionary mapping dims to lists of values. 
        """
        for dim in dims:
            if not (dim in coords):
                raise IndexError(f"dimension {dim} not found in coords")
        self.model_params.update({"dims":dims, "coords":coords})




