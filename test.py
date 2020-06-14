from nets.efficientdet import Efficientdet

model = Efficientdet(0)
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name)
