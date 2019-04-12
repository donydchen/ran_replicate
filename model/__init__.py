from .base_model import BaseModel
from .ran import RANModel



def create_model(opt):
    instance = RANModel()
    instance.initialize(opt)
    instance.setup()
    return instance

