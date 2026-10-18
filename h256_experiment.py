"""Native H256 / pre-PatchMerging encoder exchange experiment entry."""
from diffusion_experiment import main
from networks.encoder_h256 import ARCHITECTURE_ID

if __name__ == '__main__':
    main(architecture_id=ARCHITECTURE_ID)
