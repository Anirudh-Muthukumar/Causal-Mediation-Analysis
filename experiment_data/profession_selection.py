def get_profession_list():
    # Get the list of all considered professions
    # word_list = []
    with open("experiment_data/professions.json", "r") as f:
        for l in f:
            # there is only one line that eval"s to an array
            for j in eval(l):
                print(j[0], j[1], j[2])
    # return word_list

get_profession_list()