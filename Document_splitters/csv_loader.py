from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='D:\Intern\document_loaders\Social_network_ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1])
