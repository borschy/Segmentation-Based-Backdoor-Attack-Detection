from utils.data_utils import PoisonedDataset, get_raw_data
from utils.imutils import imshow

if __name__ == "__main__":
    train = get_raw_data()
    for shape in ["triangle", "L"]:
        poisoned_set = PoisonedDataset(train, 0, shape, .1)
        adv_img, adv_lbl = poisoned_set.poison_sample(1)
        imshow(adv_img, adv_lbl)
