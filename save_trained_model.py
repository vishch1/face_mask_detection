from tensorflow.keras.models import load_model


from train_mask_model import model 

# Save it
model.save("mask_detector.h5")
print("✅ Model saved as mask_detector.h5")
