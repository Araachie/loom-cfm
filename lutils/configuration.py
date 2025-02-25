import yaml

from lutils.dict_wrapper import DictWrapper


class Configuration(DictWrapper):
    """
    Represents the configuration parameters for running the process
    """

    def __init__(self, path: str):
        # Loads configuration file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        super(Configuration, self).__init__(config)

        self.check_config()

    def check_config(self):
        if self.get("training", None) is not None:
            if "source" not in self["training"]:
                self["training"]["source"] = "normal"
            if "curvature_loss" not in self["training"]["loss_weights"]:
                self["training"]["loss_weights"]["curvature_loss"] = 0.0
        if self.get("evaluation", None) is not None:
            if "source" not in self["evaluation"]:
                if self.get("training", None) is not None:
                    self["evaluation"]["source"] = self["training"]["source"]
                else:
                    self["evaluation"]["source"] = "normal"
