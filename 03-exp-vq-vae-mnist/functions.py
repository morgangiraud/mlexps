import torch


def perplexity(sample_indices, nb_classes):
    r"""
    Compute the perplexity of sample using a
    uniform distribution
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = sample_indices.view(-1, 1)
    one_hot = torch.zeros([x.shape[0], nb_classes]).to(device)
    one_hot = one_hot.scatter(1, x, 1)

    probs = torch.mean(one_hot, axis=0)
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon))

    return torch.exp(entropy)


def compute_nb_vector_used(vqvae, training_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_vector_used = torch.zeros([vqvae.quant.K]).to(device)
    for i_batch, (x, _) in enumerate(training_loader):
        x = x.to(device)

        if i_batch > 40:  # ~10000 datapoint
            break

        code = vqvae.encode(x)

        code = code.view(-1, 1)
        one_hot = torch.zeros([code.shape[0], vqvae.quant.K]).to(device)
        one_hot = one_hot.scatter(1, code, 1)
        nb_vector_used += torch.sum(one_hot, axis=0)
    nb_vector_used = torch.sum(nb_vector_used > 1)

    return nb_vector_used


if __name__ == "__main__":
    nb_classes = 10

    sample1 = torch.randint(0, 10, [11, 4, 4, 1])
    sample2 = torch.randint(0, 7, [11, 4, 4, 1])
    sample3 = torch.randint(0, 1, [11, 4, 4, 1])

    p1 = perplexity(sample1, nb_classes)
    p2 = perplexity(sample2, nb_classes)
    p3 = perplexity(sample3, nb_classes)

    print("p1: {} > p2: {} > p3: {}".format(p1, p2, p3))
    assert p2 < p1
    assert p3 < p2
