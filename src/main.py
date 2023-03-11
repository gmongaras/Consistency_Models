from score_sde.models.ncsnpp import NCSNpp
from ncspp_configs import get_ncsnpp_configs
import torch






def main():
    # Get the configuration file for the ncsp++ model
    config = get_ncsnpp_configs()

    # Create the nscp++ model
    model = NCSNpp(config)

    out = model(torch.rand((1, 3,64, 64)), torch.tensor([1]))
    print()







if __name__ == "__main__":
    main()