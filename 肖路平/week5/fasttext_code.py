import fasttext

model = fasttext.train_supervised('e:\\workspacepython\\AiPremiumClass\\cooking.stackexchange.txt',epoch=25,dim=200)

print(model.predict("What is the difference between a good and a great cook?"))
