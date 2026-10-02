import onnxruntime as ort

MODEL_PATH = "MiniFASNetV2.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("================================")
print("MODEL INFORMATION")
print("================================")

print("Inputs:")

for inp in session.get_inputs():
    print("Name  :", inp.name)
    print("Shape :", inp.shape)
    print("Type  :", inp.type)


print("\nOutputs:")

for out in session.get_outputs():
    print("Name  :", out.name)
    print("Shape :", out.shape)
    print("Type  :", out.type)