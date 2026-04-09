# config.py
import argparse
import os
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="Lancia attacchi adversarial selezionabili")
    parser.add_argument("--bim", action="store_true", help="Esegui attacco BIM")
    parser.add_argument("--fgsm", action="store_true", help="Esegui attacco FGSM")
    parser.add_argument("--pgd", action="store_true", help="Esegui attacco PGD")
    parser.add_argument("--cw", action="store_true", help="Esegui attacco Carlini & Wagner")
    parser.add_argument("--deepfool", action="store_true", help="Esegui attacco DeepFool")
    parser.add_argument("--type", choices=["generic", "targeted"], default="generic", help="Tipo di attacco")
    parser.add_argument("--net", required=True, choices=["nn1", "nn2"], help="Nome rete")
    return parser.parse_args()

# === Costanti comuni ===
SEED = 42
TESTSET_PATH = os.path.expanduser('~/aicyb/gruppo_8/test_set')
CSV_PATH = os.path.expanduser('~/aicyb/gruppo_8/identity_meta.csv')
LABELS_PATH = tf.keras.utils.get_file(
    'rcmalli_vggface_labels_v2.npy',
    "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy",
    cache_subdir="./"
)
WEIGHTS_PATH = 'resnet/model/resnet50_ft_weight.pkl'

# === BIM ===
OUTPUT_CSV_BIM_GENERIC = "~/aicyb/gruppo_8/nn1/bim/bim_generic/sec_results_bim_generic.csv"
SAVE_DIR_BIM_GENERIC = os.path.expanduser('~/aicyb/gruppo_8/nn1/bim/bim_generic/sec_adv_bim_generic')
OUTPUT_CSV_BIM_TARGETED = "~/aicyb/gruppo_8/nn1/bim/bim_targeted/sec_results_bim_targeted.csv"
SAVE_DIR_BIM_TARGETED = os.path.expanduser('~/aicyb/gruppo_8/nn1/bim/bim_targeted/sec_adv_bim_targeted')

# === PGD ===
OUTPUT_CSV_PGD_GENERIC = "~/aicyb/gruppo_8/nn1/pgd/pgd_generic/sec_results_pgd_generic.csv"
SAVE_DIR_PGD_GENERIC = os.path.expanduser('~/aicyb/gruppo_8/nn1/pgd/pgd_generic/sec_adv_pgd_generic')
OUTPUT_CSV_PGD_TARGETED = "~/aicyb/gruppo_8/nn1/pgd/pgd_targeted/sec_results_pgd_targeted.csv"
SAVE_DIR_PGD_TARGETED = os.path.expanduser('~/aicyb/gruppo_8/nn1/pgd/pgd_targeted/sec_adv_pgd_targeted')

# === FGSM ===
OUTPUT_CSV_FGSM_GENERIC = "~/aicyb/gruppo_8/nn1/fgsm/fgsm_generic/sec_results_fgsm_generic.csv"
SAVE_DIR_FGSM_GENERIC = os.path.expanduser('~/aicyb/gruppo_8/nn1/fgsm/fgsm_generic/sec_adv_fgsm_generic')
OUTPUT_CSV_FGSM_TARGETED = "~/aicyb/gruppo_8/nn1/fgsm/fgsm_targeted/sec_results_fgsm_targeted.csv"
SAVE_DIR_FGSM_TARGETED = os.path.expanduser('~/aicyb/gruppo_8/nn1/fgsm/fgsm_targeted/sec_adv_fgsm_targeted')

# === Carlini & Wagner ===
OUTPUT_CSV_CW_GENERIC = "~/aicyb/gruppo_8/nn1/cw/cw_generic/sec_results_cw_generic.csv"
SAVE_DIR_CW_GENERIC = os.path.expanduser('~/aicyb/gruppo_8/nn1/cw/cw_generic/sec_adv_cw_generic')
OUTPUT_CSV_CW_TARGETED = "~/aicyb/gruppo_8/nn1/cw/cw_targeted/sec_results_cw_targeted.csv"
SAVE_DIR_CW_TARGETED = os.path.expanduser('~/aicyb/gruppo_8/nn1/cw/cw_targeted/sec_adv_cw_targeted')

# === DeepFool ===
OUTPUT_CSV_DEEPFOOL_GENERIC = "~/aicyb/gruppo_8/nn1/deepfool/deepfool_generic/sec_results_deepfool_generic.csv"
SAVE_DIR_DEEPFOOL_GENERIC = os.path.expanduser('~/aicyb/gruppo_8/nn1/deepfool/deepfool_generic/sec_adv_deepfool_generic')

# === BIM ===
OUTPUT_CSV_BIM_GENERIC_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_generic_bim/metrics_bim_generic.csv"
OUTPUT_CSV_BIM_TARGETED_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_targeted_bim/metrics_bim_targeted.csv"

# === PGD ===
OUTPUT_CSV_PGD_GENERIC_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_generic_pgd/metrics_pgd_generic.csv"
OUTPUT_CSV_PGD_TARGETED_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_targeted_pgd/metrics_pgd_targeted.csv"

# === FGSM ===
OUTPUT_CSV_FGSM_GENERIC_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_generic_fgsm/metrics_fgsm_generic.csv"
OUTPUT_CSV_FGSM_TARGETED_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_targeted_fgsm/metrics_fgsm_targeted.csv"

# === Carlini & Wagner ===
OUTPUT_CSV_CW_GENERIC_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_generic_cw/metrics_cw_generic.csv"
OUTPUT_CSV_CW_TARGETED_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_targeted_cw/metrics_cw_targeted.csv"

# === DeepFool ===
OUTPUT_CSV_DEEPFOOL_GENERIC_NN2 = "~/aicyb/gruppo_8/nn2/results_adv_resnet50_ft_generic_deepfool/metrics_deepfool_generic.csv"
