import vietnamesetokenizer

test1= vietnamesetokenizer.VietnameseGpt2Tokenizer()

encoded=test1.encode("DANH SÁCH ĐĂNG KÝ THAM DỰ ĐÁM CƯỚI ")
decoded= test1.decode(encoded)

print (decoded)