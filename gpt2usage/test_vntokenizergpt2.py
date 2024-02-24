import vietnamesetokenizer

test1= vietnamesetokenizer.VietnameseGpt2Tokenizer()

encoded=test1.encode("Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư")
decoded= test1.decode(encoded)

print (decoded)