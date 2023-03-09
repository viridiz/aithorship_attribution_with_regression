from utils import *


# nevermind these try-excepts
# it's just a pesky feature of my machine
authors = os.listdir('train')
try:
    authors.remove('.DS_Store')
except:
    pass


idx2name = {}
for i in range(len(authors)):
    idx2name[i] = authors[i]


# list of the test books' names
test_data = os.listdir('test')
try:
    test_data.remove('.DS_Store')
except:
    pass


# train_data = [auth_1, auth_1, ...]
#   auth_j = [book1, book2, ...]
train_data = []

for auth in authors:
    train_books = os.listdir('train/' + auth)
    try:
        train_books.remove('.DS_Store')
    except:
        pass
    train_data.append(train_books)


# reg_estimates_train = [re_auth_1, ...]
#   re_auth_j = [np.matrix([[intercept_1], [slope_1]]), ...]
reg_estimates_train = []
for it in range(len(authors)):
    re_auth = []
    books_titles = train_data[it]
    for book in books_titles:
        list_of_fragments = extract_fragments('train/' + authors[it] + '/' + book)
        list_of_compressed_fragments = [LZ(it) for it in list_of_fragments]
        x = np.array(str_to_len(list_of_fragments, 1024))
        y = np.array(str_to_len(list_of_compressed_fragments))
        intercept, slope = reg_estimates(x, y)
        vec = np.matrix([[intercept], [slope]])
        re_auth.append(vec)
    reg_estimates_train.append(re_auth)


for test_book in test_data:
    list_of_fragments = extract_fragments('test/' + test_book)
    list_of_compressed_fragments = [LZ(it) for it in list_of_fragments]
    x = np.array(str_to_len(list_of_fragments, 1024))
    y = np.array(str_to_len(list_of_compressed_fragments))
    intercept, slope = reg_estimates(x, y)
    vec = np.matrix([[intercept], [slope]])
    S = cov_matrix(x, y)
    # distances = [auth_1_d, ...]
    # auth_j_d = [distances between test_book and each of the author's train books]
    distances = []
    for re_auth in reg_estimates_train:
        auth_d = []
        for re in re_auth:
            dist = distance(vec, re, S)
            auth_d.append(dist)
        distances.append(auth_d)
    max_dist = [max(it) for it in distances]
    idx_minmax = max_dist.index(min(max_dist))
    print(test_book, 'Predicted author: ' + idx2name[idx_minmax])

"""
Output:
Melville Herman. Bartleby, The Scrivener A Story of Wall-Street.txt Predicted author: melville
Dickens Charles. Hard Times.txt Predicted author: melville
Melville Herman. Redburn. His First Voyage.txt Predicted author: melville
Melville Herman. Typee A Romance of the South Sea.txt Predicted author: melville
Dickens Charles. Oliver Twist.txt Predicted author: melville
Dickens Charles. Great Expectations.txt Predicted author: dickens
"""
