from rake_nltk import Rake
import nltk
import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('punkt_tab')
# nltk.download('stopwords')

r = Rake()

text = ("Inverting OII 83.4 nm Dayglow Profiles Using Markov Chain Radiative Transfer"
        "Emission profiles of the resonantly scattered OII~83.4~nm triplet can in\nprinciple be used to estimate \\(\\mathrm{O}^+\\) "
        "density profiles in the F2\nregion of the ionosphere. Given the emission source profile, solution of this\ninverse problem"
        " is possible, but requires significant computation. The\ntraditional Feautrier solution to the radiative transfer problem "
        "requires many\niterations to converge, making it time-consuming to compute. A Markov chain\napproach to the problem produces similar "
        "results by directly constructing a\nmatrix that maps the source emission rate to an effective emission rate which\nincludes scattering"
        " to all orders. The Markov chain approach presented here\nyields faster results and therefore can be used to perform the \\(\\mathrm{O}^+\\)\ndensity "
        "retrieval with higher resolution than would otherwise be possible.")

# Extraction given the text.
r.extract_keywords_from_text(text)

# Extraction given the list of strings where each string is a sentence.
# r.extract_keywords_from_sentences(<list of sentences>)

# To get keyword phrases ranked highest to lowest.
# print(r.get_ranked_phrases())

# To get keyword phrases ranked highest to lowest with scores.
print(r.get_ranked_phrases_with_scores())
