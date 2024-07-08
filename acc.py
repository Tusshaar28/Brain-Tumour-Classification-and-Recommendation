from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator as im

testdata = im(rescale=1./255)

testsplit = testdata.flow_from_directory("Testing",target_size=(128,128),seed=123,batch_size=16,shuffle=False)

model = load_model("braintumourclassifier3rd.h5")
model.evaluate(testsplit)
print(model.summary())