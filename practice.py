import torch


if __name__ == "__main__":
    node1 = "/mydata/flcode/models/node1.pkl"
    sm1 = torch.load(node1)

    node0 = "/mydata/flcode/models/node1.pkl"
    sm0 = torch.load(node0)
    print(len(sm0[0]))

    print(len(sm1[0]))


    sm3 = sm1 + sm0
    print(len(sm3[1]))