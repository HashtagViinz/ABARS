from cv.albumentations_hook import inject_custom_albumentations
import ultralytics.data.augment as aug

inject_custom_albumentations("grayscale")
obj = aug.Albumentations()
print("Transforms in object:")
print(obj.transform)
