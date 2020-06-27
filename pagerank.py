import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    ret = dict()
    # Define the constant that is added to all probabilities
    added_constant = 1 - damping_factor

    # If the page has no outgoing links, then the transition models has all pages with equal probability
    no_of_pages = len(corpus.keys())
    if len(corpus[page]) == 0:
        for single_page in corpus.keys():
            ret[single_page] = 1 / no_of_pages
        return ret

    # The probability of each page that the current page has no link to is  (1 - damping_factor) / N
    for single_page in corpus.keys():
        if single_page not in corpus[page]:
            ret[single_page] = added_constant / no_of_pages

    # The probability of each page that the current page has a link to is (damping factor / no. of links ) + (1 - d) / N
    for next_page in corpus[page]:
        ret[next_page] = (damping_factor / len(corpus[page])) + (added_constant / no_of_pages)

    return ret


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ret = dict()

    for page in corpus.keys():
        ret[page] = 0

    # The first page is chosen randomly
    first_page = random.choice(list(corpus.keys()))

    ret[first_page] += 1

    previous_model = transition_model(corpus, first_page, damping_factor)
    # In each sample, a page is chosen according to the probabilites from the previos model with their values 
    for i in range(1, n):
        # random.choices() is used to select the page randomly from the previous model with its corresponding probability
        seleced_page = random.choices(list(previous_model.keys()), list(previous_model.values()))[0]
        # Increasing the number of occurences of the selected page in all samples
        ret[seleced_page] += 1
        # Updating the previous model for the next iteration with the model of the selected page
        previous_model = transition_model(corpus, seleced_page, damping_factor)

    # Finally, the page rank is the number of occurences of a page in all samples divided by the no of the samples
    for page in ret.keys():
        ret[page] = ret[page] / n

    return ret


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ret = dict()
    # Define the constant that is added to all probabilities
    no_of_pages = len(corpus.keys())
    added_constant = (1 - damping_factor) / no_of_pages

    # At first, all pages have equal probabilty which is 1 / all pages 
    for page in corpus.keys():
        ret[page] = 1 / no_of_pages

    iter = 0
    # Loop until the margin of error is achieved
    while(True):
        iter += 1
        # A temporary dictionary to hold all new values for page ranks
        tmpdict = dict()
        for page in corpus.keys():
            tmpdict[page] = ret[page]
        # Update the value for each page
        for p in corpus.keys():
            total = 0
            for i in corpus.keys():
                # If a page has no links, then it is considered to have links to all pages
                if len(corpus[i]) == 0:
                    total += ret[i] / no_of_pages
                elif p in corpus[i]:
                    total += ret[i] / len(corpus[i])
            tmpdict[p] = added_constant + (damping_factor * total)

        mxv = -1
        # Update the final dictionary
        for page in corpus.keys():
            mxv = max(mxv, abs(ret[page] - tmpdict[page]))
            ret[page] = tmpdict[page]

        if mxv < 0.001:
            break
    # print("Num of Iterations:", iter)
    return ret


if __name__ == "__main__":
    main()
