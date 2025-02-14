from dataclasses import dataclass

@dataclass(frozen=True)
class Feature:
    """
    Data structure for storing feature metadata.
    
    Attributes:
      name: The name of the feature (used as the output column name).
      scope: The level at which the feature operates ('session' or 'user').
      categorical: Boolean flag; True if the feature output is categorical.
    """
    name: str
    scope: str = 'session'
    categorical: bool = False
    threshold: bool = False
    
    @property
    def dtype(self) -> str:
        """
        Determines the target dtype for the feature.
        Categorical features are stored as Pandas' string dtype,
        and non-categorical features as int.
        """
        return "string" if self.categorical else "int"
    