from pseudomultitasknet import PseudoMultiTaskNet
from revnet import revnet18, revnet110


def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


# print("revnet38: {}".format(get_param_size(models.revnet38())))
# print("resnet32: {}".format(get_param_size(models.resnet32())))

print(f"standard: {get_param_size(PseudoMultiTaskNet())}")
print(f"revnet110: {get_param_size(revnet110())}")
print(f"small: {get_param_size(PseudoMultiTaskNet(small=True))}")
print(f"revnet18: {get_param_size(revnet18())}")
