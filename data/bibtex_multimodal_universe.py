all_dataset_names = [
    "apogee",
    "chandra",
    "des_y3_sne_ia",
    "desi",
    "desi_provabgs",
    "foundation",
    "gaia",
    "gz10",
    "hsc",
    "jwst",
    "legacysurvey",
    "plasticc",
    "ps1_sne_ia",
    "sdss",
    "snls",
    "ssl_legacysurvey",
    "swift_sne_ia",
    "tess",
    "vipers",
    "yse",
]

citation_instructions = {
    "apogee": """
        From https://www.sdss4.org/collaboration/citing-sdss/: 

        In addition, the appropriate SDSS acknowledgment(s) for the survey and data releases that were used should be included in the Acknowledgments section: 

        Funding for the Sloan Digital Sky 
        Survey IV has been provided by the 
        Alfred P. Sloan Foundation, the U.S. 
        Department of Energy Office of 
        Science, and the Participating 
        Institutions. 

        SDSS-IV acknowledges support and 
        resources from the Center for High 
        Performance Computing  at the 
        University of Utah. The SDSS 
        website is www.sdss4.org.

        SDSS-IV is managed by the 
        Astrophysical Research Consortium 
        for the Participating Institutions 
        of the SDSS Collaboration including 
        the Brazilian Participation Group, 
        the Carnegie Institution for Science, 
        Carnegie Mellon University, Center for 
        Astrophysics | Harvard \& 
        Smithsonian, the Chilean Participation 
        Group, the French Participation Group, 
        Instituto de Astrof\'isica de 
        Canarias, The Johns Hopkins 
        University, Kavli Institute for the 
        Physics and Mathematics of the 
        Universe (IPMU) / University of 
        Tokyo, the Korean Participation Group, 
        Lawrence Berkeley National Laboratory, 
        Leibniz Institut f\"ur Astrophysik 
        Potsdam (AIP),  Max-Planck-Institut 
        f\"ur Astronomie (MPIA Heidelberg), 
        Max-Planck-Institut f\"ur 
        Astrophysik (MPA Garching), 
        Max-Planck-Institut f\"ur 
        Extraterrestrische Physik (MPE), 
        National Astronomical Observatories of 
        China, New Mexico State University, 
        New York University, University of 
        Notre Dame, Observat\'ario 
        Nacional / MCTI, The Ohio State 
        University, Pennsylvania State 
        University, Shanghai 
        Astronomical Observatory, United 
        Kingdom Participation Group, 
        Universidad Nacional Aut\'onoma 
        de M\'exico, University of Arizona, 
        University of Colorado Boulder, 
        University of Oxford, University of 
        Portsmouth, University of Utah, 
        University of Virginia, University 
        of Washington, University of 
        Wisconsin, Vanderbilt University, 
        and Yale University.
        """,
    "chandra": """
        From https://cxc.cfa.harvard.edu/csc/cite.html :

        Users are kindly requested to acknowledge their use of the Chandra Source Catalog in any resulting publications.

        This will help us greatly to keep track of catalog usage, information that is essential for providing full accountability of our work and services, as well as for planning future services.

        The following language is suggested:

        This research has made use of data obtained from the Chandra Source Catalog, provided by the Chandra X-ray Center (CXC) as part of the Chandra Data Archive.

        """,
    "des_y3_sne_ia": """
        From https://www.darkenergysurvey.org/the-des-project/data-access/ : 

        We request that all papers that use DES public data include the acknowledgement below. In addition, we would appreciate if authors of all such papers would cite the following papers where appropriate:

        DR1
        The Dark Energy Survey Data Release 1, DES Collaboration (2018)
        The Dark Energy Survey Image Processing Pipeline, E. Morganson, et al. (2018)
        The Dark Energy Camera, B. Flaugher, et al, AJ, 150, 150 (2015)

        This project used public archival data from the Dark Energy Survey (DES). Funding for the DES Projects has been provided by the U.S. Department of Energy, the U.S. National Science Foundation, the Ministry of Science and Education of Spain, the Science and Technology FacilitiesCouncil of the United Kingdom, the Higher Education Funding Council for England, the National Center for Supercomputing Applications at the University of Illinois at Urbana-Champaign, the Kavli Institute of Cosmological Physics at the University of Chicago, the Center for Cosmology and Astro-Particle Physics at the Ohio State University, the Mitchell Institute for Fundamental Physics and Astronomy at Texas A\&M University, Financiadora de Estudos e Projetos, Funda{\c c}{\~a}o Carlos Chagas Filho de Amparo {\`a} Pesquisa do Estado do Rio de Janeiro, Conselho Nacional de Desenvolvimento Cient{\'i}fico e Tecnol{\'o}gico and the Minist{\'e}rio da Ci{\^e}ncia, Tecnologia e Inova{\c c}{\~a}o, the Deutsche Forschungsgemeinschaft, and the Collaborating Institutions in the Dark Energy Survey.

        The Collaborating Institutions are Argonne National Laboratory, the University of California at Santa Cruz, the University of Cambridge, Centro de Investigaciones Energ{\'e}ticas, Medioambientales y Tecnol{\'o}gicas-Madrid, the University of Chicago, University College London, the DES-Brazil Consortium, the University of Edinburgh, the Eidgen{\"o}ssische Technische Hochschule (ETH) Z{\"u}rich,  Fermi National Accelerator Laboratory, the University of Illinois at Urbana-Champaign, the Institut de Ci{\`e}ncies de l'Espai (IEEC/CSIC), the Institut de F{\'i}sica d'Altes Energies, Lawrence Berkeley National Laboratory, the Ludwig-Maximilians Universit{\"a}t M{\"u}nchen and the associated Excellence Cluster Universe, the University of Michigan, the National Optical Astronomy Observatory, the University of Nottingham, The Ohio State University, the OzDES Membership Consortium, the University of Pennsylvania, the University of Portsmouth, SLAC National Accelerator Laboratory, Stanford University, the University of Sussex, and Texas A\&M University.

        Based in part on observations at Cerro Tololo Inter-American Observatory, National Optical Astronomy Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA) under a cooperative agreement with the National Science Foundation.
        """,
    "desi": """
        From https://data.desi.lbl.gov/doc/acknowledgments/ : 

        The Dark Energy Spectroscopic Instrument (DESI) data are licensed under the Creative Commons Attribution 4.0 International License (“CC BY 4.0”, Summary, Full Legal Code). Users are free to share, copy, redistribute, adapt, transform and build upon the DESI data available through this website for any purpose, including commercially.

        When DESI data are used, the appropriate credit is required by using both the following reference and acknowledgments text.

        If you are using DESI data, you must cite the following reference and clearly state any changes made to these data:

        DESI Collaboration et al. (2023b), “The Early Data Release of the Dark Energy Spectroscopic Instrument”

        Also consider citing publications from the Technical Papers section if they cover any material used in your work.

        As part of the license attributes, you are obliged to include the following acknowledgments:

        This research used data obtained with the Dark Energy Spectroscopic Instrument (DESI). DESI construction and operations is managed by the Lawrence Berkeley National Laboratory. This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of High-Energy Physics, under Contract No. DE–AC02–05CH11231, and by the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility under the same contract. Additional support for DESI was provided by the U.S. National Science Foundation (NSF), Division of Astronomical Sciences under Contract No. AST-0950945 to the NSF’s National Optical-Infrared Astronomy Research Laboratory; the Science and Technology Facilities Council of the United Kingdom; the Gordon and Betty Moore Foundation; the Heising-Simons Foundation; the French Alternative Energies and Atomic Energy Commission (CEA); the National Council of Science and Technology of Mexico (CONACYT); the Ministry of Science and Innovation of Spain (MICINN), and by the DESI Member Institutions: www.desi.lbl.gov/collaborating-institutions. The DESI collaboration is honored to be permitted to conduct scientific research on Iolkam Du’ag (Kitt Peak), a mountain with particular significance to the Tohono O’odham Nation. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. National Science Foundation, the U.S. Department of Energy, or any of the listed funding agencies.
        """,
    "desi_provabgs": """
        From https://github.com/changhoonhahn/provabgs and https://arxiv.org/abs/2202.01809 :

        This research is supported by the Director, Office of Science, Office of High Energy Physics of the U.S. Department of Energy under Contract No. DE–AC02–05CH11231, and by the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility under the same contract; additional support for DESI is provided by the U.S. National Science Foundation, Division of Astronomical Sciences under Contract No. AST-0950945 to the NSF’s National Optical-Infrared Astronomy Research Laboratory; the Science and Technologies Facilities Council of the United Kingdom; the Gordon and Betty Moore Foundation; the Heising-Simons Foundation; the French Alternative Energies and Atomic Energy Commission (CEA); the National Council of Science and Technology of Mexico; the Ministry of Economy of Spain, and by the DESI Member Institutions.

        The authors are honored to be permitted to conduct scientific research on Iolkam Du’ag (Kitt Peak), a mountain with particular significance to the Tohono O’odham Nation.

        """,
    "foundation": """
        When using these data, please cite:

        Foley et al. (2018) - https://ui.adsabs.harvard.edu/abs/2018MNRAS.475..193F
        Jones et al. (2019) - https://ui.adsabs.harvard.edu/abs/2019ApJ...881...19J

        Please contact David Jones with any questions. You may also raise an issue on github, github.com/djones1040/Foundation_DR1.

        Pan-STARRS is supported in part by the National Aeronautics and Space Administration under Grants
        NNX12AT65G and NNX14AM74G. The Pan-STARRS1
        Surveys (PS1) and the PS1 public science archive have
        been made possible through contributions by the Institute
        for Astronomy, the University of Hawaii, the Pan-STARRS
        Project Office, the Max-Planck Society and its participating institutes, the Max Planck Institute for Astronomy, Heidelberg and the Max Planck Institute for Extraterrestrial
        Physics, Garching, The Johns Hopkins University, Durham
        University, the University of Edinburgh, the Queen’s University Belfast, the Harvard-Smithsonian Center for Astrophysics, the Las Cumbres Observatory Global Telescope
        Network Incorporated, the National Central University of
        Taiwan, the Space Telescope Science Institute, the National
        Aeronautics and Space Administration under Grant No.
        NNX08AR22G issued through the Planetary Science Division of the NASA Science Mission Directorate, the National
        Science Foundation Grant No. AST–1238877, the University of Maryland, Eotvos Lorand University (ELTE), the
        Los Alamos National Laboratory, and the Gordon and Betty Moore Foundation.
        """,
    "gaia": r"""
        If you have used Gaia DR3 data in your research, please use the following acknowledgement:

        This work has made use of data from the European Space Agency (ESA) mission
        {\it Gaia} (\url{https://www.cosmos.esa.int/gaia}), processed by the {\it Gaia}
        Data Processing and Analysis Consortium (DPAC,
        \url{https://www.cosmos.esa.int/web/gaia/dpac/consortium}). Funding for the DPAC
        has been provided by national institutions, in particular the institutions
        participating in the {\it Gaia} Multilateral Agreement.

        """,
    "gz10": """

        The GZ10 catalog from Leung et al. (2018) is a dataset of 17,736 galaxies with labels from the Galaxy Zoo 2 project. The catalog includes the following features for each galaxy: right ascension, declination, redshift, and a label. Galaxy10 DECaLS images come from DESI Legacy Imaging Surveys and labels come from Galaxy Zoo.

        Galaxy10 dataset classification labels come from Galaxy Zoo
        Galaxy10 dataset images come from DESI Legacy Imaging Surveys

        Galaxy Zoo is described in Lintott et al. 2008, the GalaxyZoo Data Release 2 is described in Lintott et al. 2011, Galaxy Zoo DECaLS Campaign is described in Walmsley M. et al. 2021, DESI Legacy Imaging Surveys is described in Dey A. et al., 2019

        The Legacy Surveys consist of three individual and complementary projects: the Dark Energy Camera Legacy Survey (DECaLS; Proposal ID #2014B-0404; PIs: David Schlegel and Arjun Dey), the Beijing-Arizona Sky Survey (BASS; NOAO Prop. ID #2015A-0801; PIs: Zhou Xu and Xiaohui Fan), and the Mayall z-band Legacy Survey (MzLS; Prop. ID #2016A-0453; PI: Arjun Dey). DECaLS, BASS and MzLS together include data obtained, respectively, at the Blanco telescope, Cerro Tololo Inter-American Observatory, NSF’s NOIRLab; the Bok telescope, Steward Observatory, University of Arizona; and the Mayall telescope, Kitt Peak National Observatory, NOIRLab. The Legacy Surveys project is honored to be permitted to conduct astronomical research on Iolkam Du’ag (Kitt Peak), a mountain with particular significance to the Tohono O’odham Nation.

        """,
    "hsc": """
        Materials on this website, including images from hscMap, can be used without prior permission within the following scopes.

        Extent of free use stipulated by Japanese copyright law (private use, educational use, news reporting, etc.)
        Usage in academic research, education, and learning activities
        Usage by news organizations
        Usage in printed media
        Usage in websites and social networks

        In all the cases, please explicitly include the credit, “NAOJ / HSC Collaboration“.
        See also the following web page about “Guide to Using NAOJ Website Copyrighted Materials” for details.
        https://www.nao.ac.jp/en/terms/copyright.html (in English)
        """,
    "jwst": """
        About the DJA
        The Cosmic Dawn Center is involved in a number of James Webb Space Telescope (JWST) surveys, but the public data can also be thought of as one comprehensive survey. The DAWN JWST Archive (DJA) is a repository of public JWST galaxy data, reduced with grizli and msaexp, and released for use by anyone.

        Citing the DJA
        We kindly request all scientific papers based on data or products downloaded from the Dawn JWST Archive (DJA) to include the following acknowledgement:

        (Some of) The data products presented herein were retrieved from the Dawn JWST Archive (DJA). DJA is an initiative of the Cosmic Dawn Center (DAWN), which is funded by the Danish National Research Foundation under grant DNRF140.

        """,
    "legacysurvey": """
        Data Release 10 (DR10) is the tenth public data release of the Legacy Surveys.

        When using data from the Legacy Surveys in papers, please use the following acknowledgment:

        The Legacy Surveys consist of three individual and complementary projects: the Dark Energy Camera Legacy Survey (DECaLS; Proposal ID #2014B-0404; PIs: David Schlegel and Arjun Dey), the Beijing-Arizona Sky Survey (BASS; NOAO Prop. ID #2015A-0801; PIs: Zhou Xu and Xiaohui Fan), and the Mayall z-band Legacy Survey (MzLS; Prop. ID #2016A-0453; PI: Arjun Dey). DECaLS, BASS and MzLS together include data obtained, respectively, at the Blanco telescope, Cerro Tololo Inter-American Observatory, NSF’s NOIRLab; the Bok telescope, Steward Observatory, University of Arizona; and the Mayall telescope, Kitt Peak National Observatory, NOIRLab. Pipeline processing and analyses of the data were supported by NOIRLab and the Lawrence Berkeley National Laboratory (LBNL). The Legacy Surveys project is honored to be permitted to conduct astronomical research on Iolkam Du’ag (Kitt Peak), a mountain with particular significance to the Tohono O’odham Nation.

        NOIRLab is operated by the Association of Universities for Research in Astronomy (AURA) under a cooperative agreement with the National Science Foundation. LBNL is managed by the Regents of the University of California under contract to the U.S. Department of Energy.

        This project used data obtained with the Dark Energy Camera (DECam), which was constructed by the Dark Energy Survey (DES) collaboration. Funding for the DES Projects has been provided by the U.S. Department of Energy, the U.S. National Science Foundation, the Ministry of Science and Education of Spain, the Science and Technology Facilities Council of the United Kingdom, the Higher Education Funding Council for England, the National Center for Supercomputing Applications at the University of Illinois at Urbana-Champaign, the Kavli Institute of Cosmological Physics at the University of Chicago, Center for Cosmology and Astro-Particle Physics at the Ohio State University, the Mitchell Institute for Fundamental Physics and Astronomy at Texas A&M University, Financiadora de Estudos e Projetos, Fundacao Carlos Chagas Filho de Amparo, Financiadora de Estudos e Projetos, Fundacao Carlos Chagas Filho de Amparo a Pesquisa do Estado do Rio de Janeiro, Conselho Nacional de Desenvolvimento Cientifico e Tecnologico and the Ministerio da Ciencia, Tecnologia e Inovacao, the Deutsche Forschungsgemeinschaft and the Collaborating Institutions in the Dark Energy Survey. The Collaborating Institutions are Argonne National Laboratory, the University of California at Santa Cruz, the University of Cambridge, Centro de Investigaciones Energeticas, Medioambientales y Tecnologicas-Madrid, the University of Chicago, University College London, the DES-Brazil Consortium, the University of Edinburgh, the Eidgenossische Technische Hochschule (ETH) Zurich, Fermi National Accelerator Laboratory, the University of Illinois at Urbana-Champaign, the Institut de Ciencies de l’Espai (IEEC/CSIC), the Institut de Fisica d’Altes Energies, Lawrence Berkeley National Laboratory, the Ludwig Maximilians Universitat Munchen and the associated Excellence Cluster Universe, the University of Michigan, NSF’s NOIRLab, the University of Nottingham, the Ohio State University, the University of Pennsylvania, the University of Portsmouth, SLAC National Accelerator Laboratory, Stanford University, the University of Sussex, and Texas A&M University.

        BASS is a key project of the Telescope Access Program (TAP), which has been funded by the National Astronomical Observatories of China, the Chinese Academy of Sciences (the Strategic Priority Research Program “The Emergence of Cosmological Structures” Grant # XDB09000000), and the Special Fund for Astronomy from the Ministry of Finance. The BASS is also supported by the External Cooperation Program of Chinese Academy of Sciences (Grant # 114A11KYSB20160057), and Chinese National Natural Science Foundation (Grant # 12120101003, # 11433005).

        The Legacy Survey team makes use of data products from the Near-Earth Object Wide-field Infrared Survey Explorer (NEOWISE), which is a project of the Jet Propulsion Laboratory/California Institute of Technology. NEOWISE is funded by the National Aeronautics and Space Administration.

        The Legacy Surveys imaging of the DESI footprint is supported by the Director, Office of Science, Office of High Energy Physics of the U.S. Department of Energy under Contract No. DE-AC02-05CH1123, by the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility under the same contract; and by the U.S. National Science Foundation, Division of Astronomical Sciences under Contract No. AST-0950945 to NOAO.

        """,
    "plasticc": """
        CC BY 4.0

        Acknowledgement:

        PLAsTiCC is funded through LSST Corporation Grant Award # 2017-03 and administered by the University of Toronto. Financial support for LSST comes from the National Science Foundation (NSF) through Cooperative Agreement No. 1258333, the Department of Energy (DOE) Office of Science under Contract No. DE-AC02-76SF00515, and private funding raised by the LSST Corporation. The NSF-funded LSST Project Office for construction was established as an operating center under management of the Association of Universities for Research in Astronomy (AURA). The DOE-funded effort to build the LSST camera is managed by the SLAC National Accelerator Laboratory (SLAC).

        The National Science Foundation (NSF) is an independent federal agency created by Congress in 1950 to promote the progress of science. NSF supports basic research and people to create knowledge that transforms the future.
        """,
    "ps1_sne_ia": """
        Here is the text for acknowledging PS1 in your publications:

        The Pan-STARRS1 Surveys (PS1) and the PS1 public science archive have been made possible through contributions by the Institute for Astronomy, the University of Hawaii, the Pan-STARRS Project Office, the Max-Planck Society and its participating institutes, the Max Planck Institute for Astronomy, Heidelberg and the Max Planck Institute for Extraterrestrial Physics, Garching, The Johns Hopkins University, Durham University, the University of Edinburgh, the Queen's University Belfast, the Harvard-Smithsonian Center for Astrophysics, the Las Cumbres Observatory Global Telescope Network Incorporated, the National Central University of Taiwan, the Space Telescope Science Institute, the National Aeronautics and Space Administration under Grant No. NNX08AR22G issued through the Planetary Science Division of the NASA Science Mission Directorate, the National Science Foundation Grant No. AST-1238877, the University of Maryland, Eotvos Lorand University (ELTE), the Los Alamos National Laboratory, and the Gordon and Betty Moore Foundation.

        In addition, please cite the following papers describing the instrument, survey, and data analysis as appropriate:

        The Pan-STARRS1 Surveys, Chambers, K.C., et al.
        Pan-STARRS Data Processing System, Magnier, E. A., et al.
        Pan-STARRS Pixel Processing: Detrending, Warping, Stacking, Waters, C. Z., et al.
        Pan-STARRS Pixel Analysis: Source Detection and Characterization, Magnier, E. A., et al.
        Pan-STARRS Photometric and Astrometric Calibration, Magnier, E. A., et al.
        The Pan-STARRS1 Database and Data Products, Flewelling, H. A., et al.

        """,
    "sdss": """

        https://www.sdss4.org/collaboration/citing-sdss/

        Funding for the Sloan Digital Sky Survey IV has been provided by the Alfred P. Sloan Foundation, the U.S. Department of Energy Office of Science, and the Participating Institutions. SDSS acknowledges support and resources from the Center for High-Performance Computing at the University of Utah. The SDSS web site is www.sdss4.org.

        SDSS is managed by the Astrophysical Research Consortium for the Participating Institutions of the SDSS Collaboration including the Brazilian Participation Group, the Carnegie Institution for Science, Carnegie Mellon University, Center for Astrophysics | Harvard & Smithsonian (CfA), the Chilean Participation Group, the French Participation Group, Instituto de Astrofísica de Canarias, The Johns Hopkins University, Kavli Institute for the Physics and Mathematics of the Universe (IPMU) / University of Tokyo, the Korean Participation Group, Lawrence Berkeley National Laboratory, Leibniz Institut für Astrophysik Potsdam (AIP), Max-Planck-Institut für Astronomie (MPIA Heidelberg), Max-Planck-Institut für Astrophysik (MPA Garching), Max-Planck-Institut für Extraterrestrische Physik (MPE), National Astronomical Observatories of China, New Mexico State University, New York University, University of Notre Dame, Observatório Nacional / MCTI, The Ohio State University, Pennsylvania State University, Shanghai Astronomical Observatory, United Kingdom Participation Group, Universidad Nacional Autónoma de México, University of Arizona, University of Colorado Boulder, University of Oxford, University of Portsmouth, University of Utah, University of Virginia, University of Washington, University of Wisconsin, Vanderbilt University, and Yale University.

        In addition, the appropriate SDSS acknowledgment(s) for the survey and data releases that were used should be included in the Acknowledgments section: 

        Funding for the Sloan Digital Sky 
        Survey IV has been provided by the 
        Alfred P. Sloan Foundation, the U.S. 
        Department of Energy Office of 
        Science, and the Participating 
        Institutions. 

        SDSS-IV acknowledges support and 
        resources from the Center for High 
        Performance Computing  at the 
        University of Utah. The SDSS 
        website is www.sdss4.org.

        SDSS-IV is managed by the 
        Astrophysical Research Consortium 
        for the Participating Institutions 
        of the SDSS Collaboration including 
        the Brazilian Participation Group, 
        the Carnegie Institution for Science, 
        Carnegie Mellon University, Center for 
        Astrophysics | Harvard \& 
        Smithsonian, the Chilean Participation 
        Group, the French Participation Group, 
        Instituto de Astrof\'isica de 
        Canarias, The Johns Hopkins 
        University, Kavli Institute for the 
        Physics and Mathematics of the 
        Universe (IPMU) / University of 
        Tokyo, the Korean Participation Group, 
        Lawrence Berkeley National Laboratory, 
        Leibniz Institut f\"ur Astrophysik 
        Potsdam (AIP),  Max-Planck-Institut 
        f\"ur Astronomie (MPIA Heidelberg), 
        Max-Planck-Institut f\"ur 
        Astrophysik (MPA Garching), 
        Max-Planck-Institut f\"ur 
        Extraterrestrische Physik (MPE), 
        National Astronomical Observatories of 
        China, New Mexico State University, 
        New York University, University of 
        Notre Dame, Observat\'ario 
        Nacional / MCTI, The Ohio State 
        University, Pennsylvania State 
        University, Shanghai 
        Astronomical Observatory, United 
        Kingdom Participation Group, 
        Universidad Nacional Aut\'onoma 
        de M\'exico, University of Arizona, 
        University of Colorado Boulder, 
        University of Oxford, University of 
        Portsmouth, University of Utah, 
        University of Virginia, University 
        of Washington, University of 
        Wisconsin, Vanderbilt University, 
        and Yale University.
        """,
    "snls": """
        The SNLS is an International Collaboration of physicists and astronomers from various institutions in Canada, EU and US.

        Institution/group representatives (Collaboration Board) are: P. Astier (IN2P3/LPNHE, Fr), S. Basa (INSU/LAM, Fr), R. Carlberg (U. Toronto, Ca), I. Hook (U. Oxford, UK), R. Pain (CNRS/IN2P3, Fr; Chair), S. Perlmutter (LBNL, US), C. Pritchet (U. Victoria, Ca) and J. Rich (CEA/DAPNIA, Fr).

        Irfu/SPP (Saclay), IN2P3/LPNHE (Jussieu), INSU/LAM (Marseille), IN2P3/CPPM (Marseille), University of Toronto (Canada), University of Victoria (Canada)

        """,
    "ssl_legacysurvey": """
        MIT License

        Copyright (c) 2022 George Stein

        https://github.com/georgestein/ssl-legacysurvey

        https://www.legacysurvey.org/

        When using data from the Legacy Surveys in papers, please use the following acknowledgment:

        The Legacy Surveys consist of three individual and complementary projects: the Dark Energy Camera Legacy Survey (DECaLS; Proposal ID #2014B-0404; PIs: David Schlegel and Arjun Dey), the Beijing-Arizona Sky Survey (BASS; NOAO Prop. ID #2015A-0801; PIs: Zhou Xu and Xiaohui Fan), and the Mayall z-band Legacy Survey (MzLS; Prop. ID #2016A-0453; PI: Arjun Dey). DECaLS, BASS and MzLS together include data obtained, respectively, at the Blanco telescope, Cerro Tololo Inter-American Observatory, NSF’s NOIRLab; the Bok telescope, Steward Observatory, University of Arizona; and the Mayall telescope, Kitt Peak National Observatory, NOIRLab. Pipeline processing and analyses of the data were supported by NOIRLab and the Lawrence Berkeley National Laboratory (LBNL). The Legacy Surveys project is honored to be permitted to conduct astronomical research on Iolkam Du’ag (Kitt Peak), a mountain with particular significance to the Tohono O’odham Nation.

        NOIRLab is operated by the Association of Universities for Research in Astronomy (AURA) under a cooperative agreement with the National Science Foundation. LBNL is managed by the Regents of the University of California under contract to the U.S. Department of Energy.

        This project used data obtained with the Dark Energy Camera (DECam), which was constructed by the Dark Energy Survey (DES) collaboration. Funding for the DES Projects has been provided by the U.S. Department of Energy, the U.S. National Science Foundation, the Ministry of Science and Education of Spain, the Science and Technology Facilities Council of the United Kingdom, the Higher Education Funding Council for England, the National Center for Supercomputing Applications at the University of Illinois at Urbana-Champaign, the Kavli Institute of Cosmological Physics at the University of Chicago, Center for Cosmology and Astro-Particle Physics at the Ohio State University, the Mitchell Institute for Fundamental Physics and Astronomy at Texas A&M University, Financiadora de Estudos e Projetos, Fundacao Carlos Chagas Filho de Amparo, Financiadora de Estudos e Projetos, Fundacao Carlos Chagas Filho de Amparo a Pesquisa do Estado do Rio de Janeiro, Conselho Nacional de Desenvolvimento Cientifico e Tecnologico and the Ministerio da Ciencia, Tecnologia e Inovacao, the Deutsche Forschungsgemeinschaft and the Collaborating Institutions in the Dark Energy Survey. The Collaborating Institutions are Argonne National Laboratory, the University of California at Santa Cruz, the University of Cambridge, Centro de Investigaciones Energeticas, Medioambientales y Tecnologicas-Madrid, the University of Chicago, University College London, the DES-Brazil Consortium, the University of Edinburgh, the Eidgenossische Technische Hochschule (ETH) Zurich, Fermi National Accelerator Laboratory, the University of Illinois at Urbana-Champaign, the Institut de Ciencies de l’Espai (IEEC/CSIC), the Institut de Fisica d’Altes Energies, Lawrence Berkeley National Laboratory, the Ludwig Maximilians Universitat Munchen and the associated Excellence Cluster Universe, the University of Michigan, NSF’s NOIRLab, the University of Nottingham, the Ohio State University, the University of Pennsylvania, the University of Portsmouth, SLAC National Accelerator Laboratory, Stanford University, the University of Sussex, and Texas A&M University.

        BASS is a key project of the Telescope Access Program (TAP), which has been funded by the National Astronomical Observatories of China, the Chinese Academy of Sciences (the Strategic Priority Research Program “The Emergence of Cosmological Structures” Grant # XDB09000000), and the Special Fund for Astronomy from the Ministry of Finance. The BASS is also supported by the External Cooperation Program of Chinese Academy of Sciences (Grant # 114A11KYSB20160057), and Chinese National Natural Science Foundation (Grant # 12120101003, # 11433005).

        The Legacy Survey team makes use of data products from the Near-Earth Object Wide-field Infrared Survey Explorer (NEOWISE), which is a project of the Jet Propulsion Laboratory/California Institute of Technology. NEOWISE is funded by the National Aeronautics and Space Administration.

        The Legacy Surveys imaging of the DESI footprint is supported by the Director, Office of Science, Office of High Energy Physics of the U.S. Department of Energy under Contract No. DE-AC02-05CH1123, by the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility under the same contract; and by the U.S. National Science Foundation, Division of Astronomical Sciences under Contract No. AST-0950945 to NOAO.

        """,
    "swift_sne_ia": """
        GNU LESSER GENERAL PUBLIC LICENSE

        Time-series dataset from Swift SNe Ia.

        Acknowledgments (from: https://archive.stsci.edu/prepds/sousa/)

        Suggestion of text to add to "Observations" section of papers that use SOUSA data:

        This supernova was also observed in the UV with the Ultra-Violet/Optical Telescope (UVOT; Roming et al. (2005)) on the Swift spacecraft (Gehrels et al. 2004). The UV photometry was obtained from the Swift Optical/Ultraviolet Supernova Archive (SOUSA; https://archive.stsci.edu/prepds/sousa/; Brown et al. 2014). The reduction is based on that of Brown et al. (2009), including subtraction of the host galaxy count rates and uses the revised UV zeropoints and time-dependent sensitivity from Breeveld et al. (2011).
        And in the "Acknowledgements" section:

        This work made use of Swift/UVOT data reduced by P. J. Brown and released in the Swift Optical/Ultraviolet Supernova Archive (SOUSA). SOUSA is supported by NASA's Astrophysics Data Analysis Program through grant NNX13AF35G.
        """,
    "tess": """
        From: https://archive.stsci.edu/hlsp/tess-spoc 

        Citations
        Please remember to cite the appropriate paper(s) below and the DOI if you use these data in a published work. 

        Note: These HLSP data products are licensed for use under CC BY 4.0.

        References
        Caldwell et al. 2020
        Research Note describing the TESS-SPOC light curves and how they are created.
        """,
    "vipers": """
        From: http://www.vipers.inaf.it/ 

        Acknowledging VIPERS

        We kindly request all papers using VIPERS data to add the following text to their acknowledgment section: This paper uses data from the VIMOS Public Extragalactic Redshift Survey (VIPERS). VIPERS has been performed using the ESO Very Large Telescope, under the "Large Programme" 182.A-0886. The participating institutions and funding agencies are listed at http://vipers.inaf.it
        """,
    "yse": """

        CC BY 4.0

        Time-series dataset from the Young Supernova Experiment Data Release 1 (YSE DR1).

        YSE is a collaboration between the DARK Cosmology Centre (University of Copenhagen), UC Santa Cruz, the University of Illinois, and PIs Vivienne Baldassare (Washington State University), Maria Drout (University of Toronto), Kaisey Mandel (Cambridge University), Raffaella Margutti (UC Berkeley) and V. Ashley Villar (Penn State).

        From: https://yse.ucsc.edu/acknowledgements/ 

        The Young Supernova Experiment is supported by the National Science Foundation through grants AST-1518052, AST-1815935, AST-1852393, AST-1911206, AST-1909796, and AST-1944985; the David and Lucile Packard Foundation; the Gordon & Betty Moore Foundation; the Heising-Simons Foundation; NASA through grants NNG17PX03C, 80NSSC19K1386, and 80NSSC20K0953; the Danish National Research Foundation through grant DNRF132; VILLUM FONDEN Investigator grants 16599, 10123 and 25501; the Science and Technology Facilities Council through grants ST/P000312/1, ST/S006109/1 and ST/T000198/1; the Australian Research Council Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D) through project number CE170100013; the Hong Kong government through GRF grant HKU27305119; the Independent Research Fund Denmark via grant numbers DFF 4002-00275 and 8021-00130, and the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie through grant No. 891744.

        The Pan-STARRS1 Surveys (PS1) and the PS1 public science archive have been made possible through contributions by the Institute for Astronomy, the University of Hawaii, the Pan-STARRS Project Office, the Max-Planck Society and its participating institutes, the Max Planck Institute for Astronomy, Heidelberg and the Max Planck Institute for Extraterrestrial Physics, Garching, The Johns Hopkins University, Durham University, the University of Edinburgh, the Queen’s University Belfast, the Harvard-Smithsonian Center for Astrophysics, the Las Cumbres Observatory Global Telescope Network Incorporated, the National Central University of Taiwan, the Space Telescope Science Institute, the National Aeronautics and Space Administration under Grant No. NNX08AR22G issued through the Planetary Science Division of the NASA Science Mission Directorate, the National Science Foundation Grant No. AST-1238877, the University of Maryland, Eotvos Lorand University (ELTE), the Los Alamos National Laboratory, and the Gordon and Betty Moore Foundation.

        The YSE team is also thankful for observations obtained with the Samuel Oschin 48-inch Telescope at the Palomar Observatory as part of the Zwicky Transient Facility project. ZTF is supported by the National Science Foundation under Grant No. AST-1440341 and a collaboration including Caltech, IPAC, the Weizmann Institute for Science, the Oskar Klein Center at Stockholm University, the University of Maryland, the University of Washington, Deutsches Elektronen-Synchrotron and Humboldt University, Los Alamos National Laboratories, the TANGO Consortium of Taiwan, the University of Wisconsin at Milwaukee, and Lawrence Berkeley National Laboratories. Operations are conducted by COO, IPAC, and UW.

        YSE computations are aided by the University of Chicago Research Computing Center, the Illinois Campus Cluster, and facilities at the National Center for Supercomputing Applications at UIUC.
        """,
}

bibtex_entries = {
    "apogee": """ 
        @ARTICLE{2017AJ....154...28B,
           author = {{Blanton}, Michael R. and {Bershady}, Matthew A. and {Abolfathi}, Bela and {Albareti}, Franco D. and {Allende Prieto}, Carlos and {Almeida}, Andres and {Alonso-Garc{\'\i}a}, Javier and {Anders}, Friedrich and {Anderson}, Scott F. and {Andrews}, Brett and {Aquino-Ort{\'\i}z}, Erik and {Arag{\'o}n-Salamanca}, Alfonso and {Argudo-Fern{\'a}ndez}, Maria and {Armengaud}, Eric and {Aubourg}, Eric and {Avila-Reese}, Vladimir and {Badenes}, Carles and {Bailey}, Stephen and {Barger}, Kathleen A. and {Barrera-Ballesteros}, Jorge and {Bartosz}, Curtis and {Bates}, Dominic and {Baumgarten}, Falk and {Bautista}, Julian and {Beaton}, Rachael and {Beers}, Timothy C. and {Belfiore}, Francesco and {Bender}, Chad F. and {Berlind}, Andreas A. and {Bernardi}, Mariangela and {Beutler}, Florian and {Bird}, Jonathan C. and {Bizyaev}, Dmitry and {Blanc}, Guillermo A. and {Blomqvist}, Michael and {Bolton}, Adam S. and {Boquien}, M{\'e}d{\'e}ric and {Borissova}, Jura and {van den Bosch}, Remco and {Bovy}, Jo and {Brandt}, William N. and {Brinkmann}, Jonathan and {Brownstein}, Joel R. and {Bundy}, Kevin and {Burgasser}, Adam J. and {Burtin}, Etienne and {Busca}, Nicol{\'a}s G. and {Cappellari}, Michele and {Delgado Carigi}, Maria Leticia and {Carlberg}, Joleen K. and {Carnero Rosell}, Aurelio and {Carrera}, Ricardo and {Chanover}, Nancy J. and {Cherinka}, Brian and {Cheung}, Edmond and {G{\'o}mez Maqueo Chew}, Yilen and {Chiappini}, Cristina and {Choi}, Peter Doohyun and {Chojnowski}, Drew and {Chuang}, Chia-Hsun and {Chung}, Haeun and {Cirolini}, Rafael Fernando and {Clerc}, Nicolas and {Cohen}, Roger E. and {Comparat}, Johan and {da Costa}, Luiz and {Cousinou}, Marie-Claude and {Covey}, Kevin and {Crane}, Jeffrey D. and {Croft}, Rupert A.~C. and {Cruz-Gonzalez}, Irene and {Garrido Cuadra}, Daniel and {Cunha}, Katia and {Damke}, Guillermo J. and {Darling}, Jeremy and {Davies}, Roger and {Dawson}, Kyle and {de la Macorra}, Axel and {Dell'Agli}, Flavia and {De Lee}, Nathan and {Delubac}, Timoth{\'e}e and {Di Mille}, Francesco and {Diamond-Stanic}, Aleks and {Cano-D{\'\i}az}, Mariana and {Donor}, John and {Downes}, Juan Jos{\'e} and {Drory}, Niv and {du Mas des Bourboux}, H{\'e}lion and {Duckworth}, Christopher J. and {Dwelly}, Tom and {Dyer}, Jamie and {Ebelke}, Garrett and {Eigenbrot}, Arthur D. and {Eisenstein}, Daniel J. and {Emsellem}, Eric and {Eracleous}, Mike and {Escoffier}, Stephanie and {Evans}, Michael L. and {Fan}, Xiaohui and {Fern{\'a}ndez-Alvar}, Emma and {Fernandez-Trincado}, J.~G. and {Feuillet}, Diane K. and {Finoguenov}, Alexis and {Fleming}, Scott W. and {Font-Ribera}, Andreu and {Fredrickson}, Alexander and {Freischlad}, Gordon and {Frinchaboy}, Peter M. and {Fuentes}, Carla E. and {Galbany}, Llu{\'\i}s and {Garcia-Dias}, R. and {Garc{\'\i}a-Hern{\'a}ndez}, D.~A. and {Gaulme}, Patrick and {Geisler}, Doug and {Gelfand}, Joseph D. and {Gil-Mar{\'\i}n}, H{\'e}ctor and {Gillespie}, Bruce A. and {Goddard}, Daniel and {Gonzalez-Perez}, Violeta and {Grabowski}, Kathleen and {Green}, Paul J. and {Grier}, Catherine J. and {Gunn}, James E. and {Guo}, Hong and {Guy}, Julien and {Hagen}, Alex and {Hahn}, ChangHoon and {Hall}, Matthew and {Harding}, Paul and {Hasselquist}, Sten and {Hawley}, Suzanne L. and {Hearty}, Fred and {Gonzalez Hern{\'a}ndez}, Jonay I. and {Ho}, Shirley and {Hogg}, David W. and {Holley-Bockelmann}, Kelly and {Holtzman}, Jon A. and {Holzer}, Parker H. and {Huehnerhoff}, Joseph and {Hutchinson}, Timothy A. and {Hwang}, Ho Seong and {Ibarra-Medel}, H{\'e}ctor J. and {da Silva Ilha}, Gabriele and {Ivans}, Inese I. and {Ivory}, KeShawn and {Jackson}, Kelly and {Jensen}, Trey W. and {Johnson}, Jennifer A. and {Jones}, Amy and {J{\"o}nsson}, Henrik and {Jullo}, Eric and {Kamble}, Vikrant and {Kinemuchi}, Karen and {Kirkby}, David and {Kitaura}, Francisco-Shu and {Klaene}, Mark and {Knapp}, Gillian R. and {Kneib}, Jean-Paul and {Kollmeier}, Juna A. and {Lacerna}, Ivan and {Lane}, Richard R. and {Lang}, Dustin and {Law}, David R. and {Lazarz}, Daniel and {Lee}, Youngbae and {Le Goff}, Jean-Marc and {Liang}, Fu-Heng and {Li}, Cheng and {Li}, Hongyu and {Lian}, Jianhui and {Lima}, Marcos and {Lin}, Lihwai and {Lin}, Yen-Ting and {Bertran de Lis}, Sara and {Liu}, Chao and {de Icaza Lizaola}, Miguel Angel C. and {Long}, Dan and {Lucatello}, Sara and {Lundgren}, Britt and {MacDonald}, Nicholas K. and {Deconto Machado}, Alice and {MacLeod}, Chelsea L. and {Mahadevan}, Suvrath and {Geimba Maia}, Marcio Antonio and {Maiolino}, Roberto and {Majewski}, Steven R. and {Malanushenko}, Elena and {Malanushenko}, Viktor and {Manchado}, Arturo and {Mao}, Shude and {Maraston}, Claudia and {Marques-Chaves}, Rui and {Masseron}, Thomas and {Masters}, Karen L. and {McBride}, Cameron K. and {McDermid}, Richard M. and {McGrath}, Brianne and {McGreer}, Ian D. and {Medina Pe{\~n}a}, Nicol{\'a}s and {Melendez}, Matthew and {Merloni}, Andrea and {Merrifield}, Michael R. and {Meszaros}, Szabolcs and {Meza}, Andres and {Minchev}, Ivan and {Minniti}, Dante and {Miyaji}, Takamitsu and {More}, Surhud and {Mulchaey}, John and {M{\"u}ller-S{\'a}nchez}, Francisco and {Muna}, Demitri and {Munoz}, Ricardo R. and {Myers}, Adam D. and {Nair}, Preethi and {Nandra}, Kirpal and {Correa do Nascimento}, Janaina and {Negrete}, Alenka and {Ness}, Melissa and {Newman}, Jeffrey A. and {Nichol}, Robert C. and {Nidever}, David L. and {Nitschelm}, Christian and {Ntelis}, Pierros and {O'Connell}, Julia E. and {Oelkers}, Ryan J. and {Oravetz}, Audrey and {Oravetz}, Daniel and {Pace}, Zach and {Padilla}, Nelson and {Palanque-Delabrouille}, Nathalie and {Alonso Palicio}, Pedro and {Pan}, Kaike and {Parejko}, John K. and {Parikh}, Taniya and {P{\^a}ris}, Isabelle and {Park}, Changbom and {Patten}, Alim Y. and {Peirani}, Sebastien and {Pellejero-Ibanez}, Marcos and {Penny}, Samantha and {Percival}, Will J. and {Perez-Fournon}, Ismael and {Petitjean}, Patrick and {Pieri}, Matthew M. and {Pinsonneault}, Marc and {Pisani}, Alice and {Poleski}, Rados{\l}aw and {Prada}, Francisco and {Prakash}, Abhishek and {Queiroz}, Anna B{\'a}rbara de Andrade and {Raddick}, M. Jordan and {Raichoor}, Anand and {Barboza Rembold}, Sandro and {Richstein}, Hannah and {Riffel}, Rogemar A. and {Riffel}, Rog{\'e}rio and {Rix}, Hans-Walter and {Robin}, Annie C. and {Rockosi}, Constance M. and {Rodr{\'\i}guez-Torres}, Sergio and {Roman-Lopes}, A. and {Rom{\'a}n-Z{\'u}{\~n}iga}, Carlos and {Rosado}, Margarita and {Ross}, Ashley J. and {Rossi}, Graziano and {Ruan}, John and {Ruggeri}, Rossana and {Rykoff}, Eli S. and {Salazar-Albornoz}, Salvador and {Salvato}, Mara and {S{\'a}nchez}, Ariel G. and {Aguado}, D.~S. and {S{\'a}nchez-Gallego}, Jos{\'e} R. and {Santana}, Felipe A. and {Santiago}, Bas{\'\i}lio Xavier and {Sayres}, Conor and {Schiavon}, Ricardo P. and {da Silva Schimoia}, Jaderson and {Schlafly}, Edward F. and {Schlegel}, David J. and {Schneider}, Donald P. and {Schultheis}, Mathias and {Schuster}, William J. and {Schwope}, Axel and {Seo}, Hee-Jong and {Shao}, Zhengyi and {Shen}, Shiyin and {Shetrone}, Matthew and {Shull}, Michael and {Simon}, Joshua D. and {Skinner}, Danielle and {Skrutskie}, M.~F. and {Slosar}, An{\v{z}}e and {Smith}, Verne V. and {Sobeck}, Jennifer S. and {Sobreira}, Flavia and {Somers}, Garrett and {Souto}, Diogo and {Stark}, David V. and {Stassun}, Keivan and {Stauffer}, Fritz and {Steinmetz}, Matthias and {Storchi-Bergmann}, Thaisa and {Streblyanska}, Alina and {Stringfellow}, Guy S. and {Su{\'a}rez}, Genaro and {Sun}, Jing and {Suzuki}, Nao and {Szigeti}, Laszlo and {Taghizadeh-Popp}, Manuchehr and {Tang}, Baitian and {Tao}, Charling and {Tayar}, Jamie and {Tembe}, Mita and {Teske}, Johanna and {Thakar}, Aniruddha R. and {Thomas}, Daniel and {Thompson}, Benjamin A. and {Tinker}, Jeremy L. and {Tissera}, Patricia and {Tojeiro}, Rita and {Hernandez Toledo}, Hector and {de la Torre}, Sylvain and {Tremonti}, Christy and {Troup}, Nicholas W. and {Valenzuela}, Octavio and {Martinez Valpuesta}, Inma and {Vargas-Gonz{\'a}lez}, Jaime and {Vargas-Maga{\~n}a}, Mariana and {Vazquez}, Jose Alberto and {Villanova}, Sandro and {Vivek}, M. and {Vogt}, Nicole and {Wake}, David and {Walterbos}, Rene and {Wang}, Yuting and {Weaver}, Benjamin Alan and {Weijmans}, Anne-Marie and {Weinberg}, David H. and {Westfall}, Kyle B. and {Whelan}, David G. and {Wild}, Vivienne and {Wilson}, John and {Wood-Vasey}, W.~M. and {Wylezalek}, Dominika and {Xiao}, Ting and {Yan}, Renbin and {Yang}, Meng and {Ybarra}, Jason E. and {Y{\`e}che}, Christophe and {Zakamska}, Nadia and {Zamora}, Olga and {Zarrouk}, Pauline and {Zasowski}, Gail and {Zhang}, Kai and {Zhao}, Gong-Bo and {Zheng}, Zheng and {Zheng}, Zheng and {Zhou}, Xu and {Zhou}, Zhi-Min and {Zhu}, Guangtun B. and {Zoccali}, Manuela and {Zou}, Hu},
            title = "{Sloan Digital Sky Survey IV: Mapping the Milky Way, Nearby Galaxies, and the Distant Universe}",
              journal = {\aj},
             keywords = {cosmology: observations, galaxies: general, Galaxy: general, instrumentation: spectrographs, stars: general, surveys, Astrophysics - Astrophysics of Galaxies},
                 year = 2017,
                month = jul,
               volume = {154},
               number = {1},
                  eid = {28},
                pages = {28},
                  doi = {10.3847/1538-3881/aa7567},
            archivePrefix = {arXiv},
                   eprint = {1703.00052},
             primaryClass = {astro-ph.GA},
                   adsurl = {https://ui.adsabs.harvard.edu/abs/2017AJ....154...28B},
                  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }

        @ARTICLE{2022ApJS..259...35A, 
            author = {{Abdurro'uf} and et al.}, 
            title = "{The Seventeenth Data Release of the Sloan Digital Sky Surveys: Complete Release of MaNGA, MaStar, and APOGEE-2 Data}", 
            journal = {pjs}, 
            keywords = {Astronomy data acquisition, Astronomy databases, Surveys, 1860, 83, 1671, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics}, 
            year = 2022, 
            month = apr, 
            volume = {259}, 
            number = {2}, 
            eid = {35}, 
            pages = {35}, 
            doi = {10.3847/1538-4365/ac4414}, 
            archivePrefix = {arXiv}, 
            eprint = {2112.02026}, 
            primaryClass = {astro-ph.GA}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
        }

        @ARTICLE{2017AJ....154...94M,
           author = {{Majewski}, S.~R. and {Schiavon}, R.~P. and {Frinchaboy}, P.~M. and 
            {Allende Prieto}, C. and {Barkhouser}, R. and {Bizyaev}, D. and 
            {Blank}, B. and {Brunner}, S. and {Burton}, A. and {Carrera}, R. and 
            {Chojnowski}, S.~D. and {Cunha}, K. and {Epstein}, C. and {Fitzgerald}, G. and 
            {Garc{\'{\i}}a P{\'e}rez}, A.~E. and {Hearty}, F.~R. and {Henderson}, C. and 
            {Holtzman}, J.~A. and {Johnson}, J.~A. and {Lam}, C.~R. and 
            {Lawler}, J.~E. and {Maseman}, P. and {M{\'e}sz{\'a}ros}, S. and 
            {Nelson}, M. and {Nguyen}, D.~C. and {Nidever}, D.~L. and {Pinsonneault}, M. and 
            {Shetrone}, M. and {Smee}, S. and {Smith}, V.~V. and {Stolberg}, T. and 
            {Skrutskie}, M.~F. and {Walker}, E. and {Wilson}, J.~C. and 
            {Zasowski}, G. and {Anders}, F. and {Basu}, S. and {Beland}, S. and 
            {Blanton}, M.~R. and {Bovy}, J. and {Brownstein}, J.~R. and 
            {Carlberg}, J. and {Chaplin}, W. and {Chiappini}, C. and {Eisenstein}, D.~J. and 
            {Elsworth}, Y. and {Feuillet}, D. and {Fleming}, S.~W. and {Galbraith-Frew}, J. and 
            {Garc{\'{\i}}a}, R.~A. and {Garc{\'{\i}}a-Hern{\'a}ndez}, D.~A. and 
            {Gillespie}, B.~A. and {Girardi}, L. and {Gunn}, J.~E. and {Hasselquist}, S. and 
            {Hayden}, M.~R. and {Hekker}, S. and {Ivans}, I. and {Kinemuchi}, K. and 
            {Klaene}, M. and {Mahadevan}, S. and {Mathur}, S. and {Mosser}, B. and 
            {Muna}, D. and {Munn}, J.~A. and {Nichol}, R.~C. and {O'Connell}, R.~W. and 
            {Parejko}, J.~K. and {Robin}, A.~C. and {Rocha-Pinto}, H. and 
            {Schultheis}, M. and {Serenelli}, A.~M. and {Shane}, N. and 
            {Silva Aguirre}, V. and {Sobeck}, J.~S. and {Thompson}, B. and 
            {Troup}, N.~W. and {Weinberg}, D.~H. and {Zamora}, O.},
            title = "{The Apache Point Observatory Galactic Evolution Experiment (APOGEE)}",
            journal = {\aj},
            archivePrefix = "arXiv",
            eprint = {1509.05420},
            primaryClass = "astro-ph.IM",
            keywords = {Galaxy: abundances, Galaxy: evolution, Galaxy: formation, Galaxy: kinematics and dynamics, Galaxy: stellar content, Galaxy: structure},
            year = 2017,
            month = sep,
            volume = 154,
              eid = {94},
            pages = {94},
              doi = {10.3847/1538-3881/aa784d},
            adsurl = {http://adsabs.harvard.edu/abs/2017AJ....154...94M},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }

        @ARTICLE{2019PASP..131e5001W,
               author = {{Wilson}, J.~C. and {Hearty}, F.~R. and {Skrutskie}, M.~F. and {Majewski}, S.~R. and {Holtzman}, J.~A. and {Eisenstein}, D. and {Gunn}, J. and {Blank}, B. and {Henderson}, C. and {Smee}, S. and {Nelson}, M. and {Nidever}, D. and {Arns}, J. and {Barkhouser}, R. and {Barr}, J. and {Beland}, S. and {Bershady}, M.~A. and {Blanton}, M.~R. and {Brunner}, S. and {Burton}, A. and {Carey}, L. and {Carr}, M. and {Colque}, J.~P. and {Crane}, J. and {Damke}, G.~J. and {Davidson}, J.~W., Jr. and {Dean}, J. and {Di Mille}, F. and {Don}, K.~W. and {Ebelke}, G. and {Evans}, M. and {Fitzgerald}, G. and {Gillespie}, B. and {Hall}, M. and {Harding}, A. and {Harding}, P. and {Hammond}, R. and {Hancock}, D. and {Harrison}, C. and {Hope}, S. and {Horne}, T. and {Karakla}, J. and {Lam}, C. and {Leger}, F. and {MacDonald}, N. and {Maseman}, P. and {Matsunari}, J. and {Melton}, S. and {Mitcheltree}, T. and {O'Brien}, T. and {O'Connell}, R.~W. and {Patten}, A. and {Richardson}, W. and {Rieke}, G. and {Rieke}, M. and {Roman-Lopes}, A. and {Schiavon}, R.~P. and {Sobeck}, J.~S. and {Stolberg}, T. and {Stoll}, R. and {Tembe}, M. and {Trujillo}, J.~D. and {Uomoto}, A. and {Vernieri}, M. and {Walker}, E. and {Weinberg}, D.~H. and {Young}, E. and {Anthony-Brumfield}, B. and {Bizyaev}, D. and {Breslauer}, B. and {De Lee}, N. and {Downey}, J. and {Halverson}, S. and {Huehnerhoff}, J. and {Klaene}, M. and {Leon}, E. and {Long}, D. and {Mahadevan}, S. and {Malanushenko}, E. and {Nguyen}, D.~C. and {Owen}, R. and {S{\'a}nchez-Gallego}, J.~R. and {Sayres}, C. and {Shane}, N. and {Shectman}, S.~A. and {Shetrone}, M. and {Skinner}, D. and {Stauffer}, F. and {Zhao}, B.},
                title = "{The Apache Point Observatory Galactic Evolution Experiment (APOGEE) Spectrographs}",
              journal = {\pasp},
             keywords = {Astrophysics - Instrumentation and Methods for Astrophysics},
                 year = 2019,
                month = may,
               volume = {131},
               number = {999},
                pages = {055001},
                  doi = {10.1088/1538-3873/ab0075},
        archivePrefix = {arXiv},
               eprint = {1902.00928},
         primaryClass = {astro-ph.IM},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2019PASP..131e5001W},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        @ARTICLE{2016AJ....151..144G, 
            author = {{Garc{'\i}a P{'e}rez}, Ana E. and {Allende Prieto}, Carlos and {Holtzman}, Jon A. and {Shetrone}, Matthew and {M{'e}sz{'a}ros}, Szabolcs and {Bizyaev}, Dmitry and {Carrera}, Ricardo and {Cunha}, Katia and {Garc{'\i}a-Hern{'a}ndez}, D.~A. and {Johnson}, Jennifer A. and {Majewski}, Steven R. and {Nidever}, David L. and {Schiavon}, Ricardo P. and {Shane}, Neville and {Smith}, Verne V. and {Sobeck}, Jennifer and {Troup}, Nicholas and {Zamora}, Olga and {Weinberg}, David H. and {Bovy}, Jo and {Eisenstein}, Daniel J. and {Feuillet}, Diane and {Frinchaboy}, Peter M. and {Hayden}, Michael R. and {Hearty}, Fred R. and {Nguyen}, Duy C. and {O'Connell}, Robert W. and {Pinsonneault}, Marc H. and {Wilson}, John C. and {Zasowski}, Gail}, 
            title = "{ASPCAP: The APOGEE Stellar Parameter and Chemical Abundances Pipeline}", 
            journal = {j}, 
            keywords = {Galaxy: center, Galaxy: structure, methods: data analysis, stars: abundances, stars: atmospheres, Astrophysics - Solar and Stellar Astrophysics}, 
            year = 2016, 
            month = jun, 
            volume = {151}, 
            number = {6}, 
            eid = {144}, 
            pages = {144}, 
            doi = {10.3847/0004-6256/151/6/144}, 
            archivePrefix = {arXiv}, 
            eprint = {1510.07635}, 
            primaryClass = {astro-ph.SR}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2016AJ....151..144G}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} }
        """,
    "chandra": """
        @ARTICLE{2010ApJS..189...37E, 
            author = {{Evans}, Ian N. and {Primini}, Francis A. and {Glotfelty}, Kenny J. and {Anderson}, Craig S. and {Bonaventura}, Nina R. and {Chen}, Judy C. and {Davis}, John E. and {Doe}, Stephen M. and {Evans}, Janet D. and {Fabbiano}, Giuseppina and {Galle}, Elizabeth C. and {Gibbs}, Danny G., II and {Grier}, John D. and {Hain}, Roger M. and {Hall}, Diane M. and {Harbo}, Peter N. and {He}, Xiangqun Helen and {Houck}, John C. and {Karovska}, Margarita and {Kashyap}, Vinay L. and {Lauer}, Jennifer and {McCollough}, Michael L. and {McDowell}, Jonathan C. and {Miller}, Joseph B. and {Mitschang}, Arik W. and {Morgan}, Douglas L. and {Mossman}, Amy E. and {Nichols}, Joy S. and {Nowak}, Michael A. and {Plummer}, David A. and {Refsdal}, Brian L. and {Rots}, Arnold H. and {Siemiginowska}, Aneta and {Sundheim}, Beth A. and {Tibbetts}, Michael S. and {Van Stone}, David W. and {Winkelman}, Sherry L. and {Zografou}, Panagoula}, 
            title = "{The Chandra Source Catalog}", 
            journal = {pjs}, 
            keywords = {catalogs, X-rays: general, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics}, 
            year = 2010, 
            month = jul, 
            volume = {189}, 
            number = {1}, 
            pages = {37-82}, 
            doi = {10.1088/0067-0049/189/1/37}, 
            archivePrefix = {arXiv}, 
            eprint = {1005.4665}, 
            primaryClass = {astro-ph.HE}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2010ApJS..189...37E}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} }
        """,
    "des_y3_sne_ia": """
        @ARTICLE{2019ApJ...874..106B, 
            author = {{Brout}, D. and {Sako}, M. and {Scolnic}, D. and {Kessler}, R. and {D'Andrea}, C.B. and {Davis}, T.M. and {Hinton}, S.R. and {Kim}, A.G. and {Lasker}, J. and {Macaulay}, E. and {M{"o}ller}, A. and {Nichol}, R.C. and {Smith}, M. and {Sullivan}, M. and {Wolf}, R.C. and {Allam}, S. and {Bassett}, B.A. and {Brown}, P. and {Castander}, F.J. and {Childress}, M. and {Foley}, R.J. and {Galbany}, L. and {Herner}, K. and {Kasai}, E. and {March}, M. and {Morganson}, E. and {Nugent}, P. and {Pan}, Y. -C. and {Thomas}, R.C. and {Tucker}, B.E. and {Wester}, W. and {Abbott}, T.M.C. and {Annis}, J. and {Avila}, S. and {Bertin}, E. and {Brooks}, D. and {Burke}, D.L. and {Carnero Rosell}, A. and {Carrasco Kind}, M. and {Carretero}, J. and {Crocce}, M. and {Cunha}, C.E. and {da Costa}, L.N. and {Davis}, C. and {De Vicente}, J. and {Desai}, S. and {Diehl}, H.T. and {Doel}, P. and {Eifler}, T.F. and {Flaugher}, B. and {Fosalba}, P. and {Frieman}, J. and {Garc{'\i}a-Bellido}, J. and {Gaztanaga}, E. and {Gerdes}, D.W. and {Goldstein}, D.A. and {Gruen}, D. and {Gruendl}, R.A. and {Gschwend}, J. and {Gutierrez}, G. and {Hartley}, W.G. and {Hollowood}, D.L. and {Honscheid}, K. and {James}, D.J. and {Kuehn}, K. and {Kuropatkin}, N. and {Lahav}, O. and {Li}, T.S. and {Lima}, M. and {Marshall}, J.L. and {Martini}, P. and {Miquel}, R. and {Nord}, B. and {Plazas}, A.A. and {Roodman}, A. and {Rykoff}, E.S. and {Sanchez}, E. and {Scarpine}, V. and {Schindler}, R. and {Schubnell}, M. and {Serrano}, S. and {Sevilla-Noarbe}, I. and {Soares-Santos}, M. and {Sobreira}, F. and {Suchyta}, E. and {Swanson}, M.E.C. and {Tarle}, G. and {Thomas}, D. and {Tucker}, D.L. and {Walker}, A.R. and {Yanny}, B. and {Zhang}, Y. and {DES COLLABORATION}}, 
            title = "{First Cosmology Results Using Type Ia Supernovae from the Dark Energy Survey: Photometric Pipeline and Light-curve Data Release}", 
            journal = {pj}, 
            keywords = {cosmology: observations, supernovae: general, techniques: photometric, Astrophysics - Instrumentation and Methods for Astrophysics}, 
            year = 2019, 
            month = mar, 
            volume = {874}, 
            number = {1}, 
            eid = {106}, 
            pages = {106}, 
            doi = {10.3847/1538-4357/ab06c1}, 
            archivePrefix = {arXiv}, 
            eprint = {1811.02378}, 
            primaryClass = {astro-ph.IM}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJ...874..106B}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
          }

        @article{Abbott_2018,
          title={The Dark Energy Survey: Data Release 1},
          volume={239},
          ISSN={1538-4365},
          url={http://dx.doi.org/10.3847/1538-4365/aae9f0},
          DOI={10.3847/1538-4365/aae9f0},
          number={2},
          journal={The Astrophysical Journal Supplement Series},
          publisher={American Astronomical Society},
          author={Abbott, T. M. C. and Abdalla, F. B. and Allam, S. and Amara, A. and Annis, J. and Asorey, J. and Avila, S. and Ballester, O. and Banerji, M. and Barkhouse, W. and Baruah, L. and Baumer, M. and Bechtol, K. and Becker, M. R. and Benoit-Lévy, A. and Bernstein, G. M. and Bertin, E. and Blazek, J. and Bocquet, S. and Brooks, D. and Brout, D. and Buckley-Geer, E. and Burke, D. L. and Busti, V. and Campisano, R. and Cardiel-Sas, L. and Rosell, A. Carnero and Kind, M. Carrasco and Carretero, J. and Castander, F. J. and Cawthon, R. and Chang, C. and Chen, X. and Conselice, C. and Costa, G. and Crocce, M. and Cunha, C. E. and D’Andrea, C. B. and Costa, L. N. da and Das, R. and Daues, G. and Davis, T. M. and Davis, C. and Vicente, J. De and DePoy, D. L. and DeRose, J. and Desai, S. and Diehl, H. T. and Dietrich, J. P. and Dodelson, S. and Doel, P. and Drlica-Wagner, A. and Eifler, T. F. and Elliott, A. E. and Evrard, A. E. and Farahi, A. and Neto, A. Fausti and Fernandez, E. and Finley, D. A. and Flaugher, B. and Foley, R. J. and Fosalba, P. and Friedel, D. N. and Frieman, J. and García-Bellido, J. and Gaztanaga, E. and Gerdes, D. W. and Giannantonio, T. and Gill, M. S. S. and Glazebrook, K. and Goldstein, D. A. and Gower, M. and Gruen, D. and Gruendl, R. A. and Gschwend, J. and Gupta, R. R. and Gutierrez, G. and Hamilton, S. and Hartley, W. G. and Hinton, S. R. and Hislop, J. M. and Hollowood, D. and Honscheid, K. and Hoyle, B. and Huterer, D. and Jain, B. and James, D. J. and Jeltema, T. and Johnson, M. W. G. and Johnson, M. D. and Kacprzak, T. and Kent, S. and Khullar, G. and Klein, M. and Kovacs, A. and Koziol, A. M. G. and Krause, E. and Kremin, A. and Kron, R. and Kuehn, K. and Kuhlmann, S. and Kuropatkin, N. and Lahav, O. and Lasker, J. and Li, T. S. and Li, R. T. and Liddle, A. R. and Lima, M. and Lin, H. and López-Reyes, P. and MacCrann, N. and Maia, M. A. G. and Maloney, J. D. and Manera, M. and March, M. and Marriner, J. and Marshall, J. L. and Martini, P. and McClintock, T. and McKay, T. and McMahon, R. G. and Melchior, P. and Menanteau, F. and Miller, C. J. and Miquel, R. and Mohr, J. J. and Morganson, E. and Mould, J. and Neilsen, E. and Nichol, R. C. and Nogueira, F. and Nord, B. and Nugent, P. and Nunes, L. and Ogando, R. L. C. and Old, L. and Pace, A. B. and Palmese, A. and Paz-Chinchón, F. and Peiris, H. V. and Percival, W. J. and Petravick, D. and Plazas, A. A. and Poh, J. and Pond, C. and Porredon, A. and Pujol, A. and Refregier, A. and Reil, K. and Ricker, P. M. and Rollins, R. P. and Romer, A. K. and Roodman, A. and Rooney, P. and Ross, A. J. and Rykoff, E. S. and Sako, M. and Sanchez, M. L. and Sanchez, E. and Santiago, B. and Saro, A. and Scarpine, V. and Scolnic, D. and Serrano, S. and Sevilla-Noarbe, I. and Sheldon, E. and Shipp, N. and Silveira, M. L. and Smith, M. and Smith, R. C. and Smith, J. A. and Soares-Santos, M. and Sobreira, F. and Song, J. and Stebbins, A. and Suchyta, E. and Sullivan, M. and Swanson, M. E. C. and Tarle, G. and Thaler, J. and Thomas, D. and Thomas, R. C. and Troxel, M. A. and Tucker, D. L. and Vikram, V. and Vivas, A. K. and Walker, A. R. and Wechsler, R. H. and Weller, J. and Wester, W. and Wolf, R. C. and Wu, H. and Yanny, B. and Zenteno, A. and Zhang, Y. and Zuntz, J. and Juneau, S. and Fitzpatrick, M. and Nikutta, R. and Nidever, D. and Olsen, K. and Scott, A.},
          year={2018},
          month=nov, pages={18} 
        }
        """,
    "desi": """
        @ARTICLE{2023arXiv230606308D,
       author = {{DESI Collaboration} and {Adame}, A.~G. and {Aguilar}, J. and {Ahlen}, S. and {Alam}, S. and {Aldering}, G. and {Alexander}, D.~M. and {Alfarsy}, R. and {Allende Prieto}, C. and {Alvarez}, M. and {Alves}, O. and {Anand}, A. and {Andrade-Oliveira}, F. and {Armengaud}, E. and {Asorey}, J. and {Avila}, S. and {Aviles}, A. and {Bailey}, S. and {Balaguera-Antol{\'\i}nez}, A. and {Ballester}, O. and {Baltay}, C. and {Bault}, A. and {Bautista}, J. and {Behera}, J. and {Beltran}, S.~F. and {BenZvi}, S. and {Beraldo e Silva}, L. and {Bermejo-Climent}, J.~R. and {Berti}, A. and {Besuner}, R. and {Beutler}, F. and {Bianchi}, D. and {Blake}, C. and {Blum}, R. and {Bolton}, A.~S. and {Brieden}, S. and {Brodzeller}, A. and {Brooks}, D. and {Brown}, Z. and {Buckley-Geer}, E. and {Burtin}, E. and {Cabayol-Garcia}, L. and {Cai}, Z. and {Canning}, R. and {Cardiel-Sas}, L. and {Carnero Rosell}, A. and {Castander}, F.~J. and {Cervantes-Cota}, J.~L. and {Chabanier}, S. and {Chaussidon}, E. and {Chaves-Montero}, J. and {Chen}, S. and {Chuang}, C. and {Claybaugh}, T. and {Cole}, S. and {Cooper}, A.~P. and {Cuceu}, A. and {Davis}, T.~M. and {Dawson}, K. and {de Belsunce}, R. and {de la Cruz}, R. and {de la Macorra}, A. and {de Mattia}, A. and {Demina}, R. and {Demirbozan}, U. and {DeRose}, J. and {Dey}, A. and {Dey}, B. and {Dhungana}, G. and {Ding}, J. and {Ding}, Z. and {Doel}, P. and {Doshi}, R. and {Douglass}, K. and {Edge}, A. and {Eftekharzadeh}, S. and {Eisenstein}, D.~J. and {Elliott}, A. and {Escoffier}, S. and {Fagrelius}, P. and {Fan}, X. and {Fanning}, K. and {Fawcett}, V.~A. and {Ferraro}, S. and {Ereza}, J. and {Flaugher}, B. and {Font-Ribera}, A. and {Forero-S{\'a}nchez}, D. and {Forero-Romero}, J.~E. and {Frenk}, C.~S. and {G{\"a}nsicke}, B.~T. and {Garc{\'\i}a}, L. {\'A}. and {Garc{\'\i}a-Bellido}, J. and {Garcia-Quintero}, C. and {Garrison}, L.~H. and {Gil-Mar{\'\i}n}, H. and {Golden-Marx}, J. and {Gontcho}, S. Gontcho A and {Gonzalez-Morales}, A.~X. and {Gonzalez-Perez}, V. and {Gordon}, C. and {Graur}, O. and {Green}, D. and {Gruen}, D. and {Guy}, J. and {Hadzhiyska}, B. and {Hahn}, C. and {Han}, J.~J. and {Hanif}, M.~M. S and {Herrera-Alcantar}, H.~K. and {Honscheid}, K. and {Hou}, J. and {Howlett}, C. and {Huterer}, D. and {Ir{\v{s}}i{\v{c}}}, V. and {Ishak}, M. and {Jacques}, A. and {Jana}, A. and {Jiang}, L. and {Jimenez}, J. and {Jing}, Y.~P. and {Joudaki}, S. and {Jullo}, E. and {Juneau}, S. and {Kizhuprakkat}, N. and {Kara{\c{c}}ayl{\i}}, N.~G. and {Karim}, T. and {Kehoe}, R. and {Kent}, S. and {Khederlarian}, A. and {Kim}, S. and {Kirkby}, D. and {Kisner}, T. and {Kitaura}, F. and {Kneib}, J. and {Koposov}, S.~E. and {Kov{\'a}cs}, A. and {Kremin}, A. and {Krolewski}, A. and {L'Huillier}, B. and {Lambert}, A. and {Lamman}, C. and {Lan}, T. -W. and {Landriau}, M. and {Lang}, D. and {Lange}, J.~U. and {Lasker}, J. and {Le Guillou}, L. and {Leauthaud}, A. and {Levi}, M.~E. and {Li}, T.~S. and {Linder}, E. and {Lyons}, A. and {Magneville}, C. and {Manera}, M. and {Manser}, C.~J. and {Margala}, D. and {Martini}, P. and {McDonald}, P. and {Medina}, G.~E. and {Medina-Varela}, L. and {Meisner}, A. and {Mena-Fern{\'a}ndez}, J. and {Meneses-Rizo}, J. and {Mezcua}, M. and {Miquel}, R. and {Montero-Camacho}, P. and {Moon}, J. and {Moore}, S. and {Moustakas}, J. and {Mueller}, E. and {Mundet}, J. and {Mu{\~n}oz-Guti{\'e}rrez}, A. and {Myers}, A.~D. and {Nadathur}, S. and {Napolitano}, L. and {Neveux}, R. and {Newman}, J.~A. and {Nie}, J. and {Nikutta}, R. and {Niz}, G. and {Norberg}, P. and {Noriega}, H.~E. and {Paillas}, E. and {Palanque-Delabrouille}, N. and {Palmese}, A. and {Zhiwei}, P. and {Parkinson}, D. and {Penmetsa}, S. and {Percival}, W.~J. and {P{\'e}rez-Fern{\'a}ndez}, A. and {P{\'e}rez-R{\`a}fols}, I. and {Pieri}, M. and {Poppett}, C. and {Porredon}, A. and {Pothier}, S. and {Prada}, F. and {Pucha}, R. and {Raichoor}, A. and {Ram{\'\i}rez-P{\'e}rez}, C. and {Ramirez-Solano}, S. and {Rashkovetskyi}, M. and {Ravoux}, C. and {Rocher}, A. and {Rockosi}, C. and {Ross}, A.~J. and {Rossi}, G. and {Ruggeri}, R. and {Ruhlmann-Kleider}, V. and {Sabiu}, C.~G. and {Said}, K. and {Saintonge}, A. and {Samushia}, L. and {Sanchez}, E. and {Saulder}, C. and {Schaan}, E. and {Schlafly}, E.~F. and {Schlegel}, D. and {Scholte}, D. and {Schubnell}, M. and {Seo}, H. and {Shafieloo}, A. and {Sharples}, R. and {Sheu}, W. and {Silber}, J. and {Sinigaglia}, F. and {Siudek}, M. and {Slepian}, Z. and {Smith}, A. and {Sprayberry}, D. and {Stephey}, L. and {Su{\'a}rez-P{\'e}rez}, J. and {Sun}, Z. and {Tan}, T. and {Tarl{\'e}}, G. and {Tojeiro}, R. and {Ure{\~n}a-L{\'o}pez}, L.~A. and {Vaisakh}, R. and {Valcin}, D. and {Valdes}, F. and {Valluri}, M. and {Vargas-Maga{\~n}a}, M. and {Variu}, A. and {Verde}, L. and {Walther}, M. and {Wang}, B. and {Wang}, M.~S. and {Weaver}, B.~A. and {Weaverdyck}, N. and {Wechsler}, R.~H. and {White}, M. and {Xie}, Y. and {Yang}, J. and {Y{\`e}che}, C. and {Yu}, J. and {Yuan}, S. and {Zhang}, H. and {Zhang}, Z. and {Zhao}, C. and {Zheng}, Z. and {Zhou}, R. and {Zhou}, Z. and {Zou}, H. and {Zou}, S. and {Zu}, Y.},
        title = "{The Early Data Release of the Dark Energy Spectroscopic Instrument}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2023,
        month = jun,
          eid = {arXiv:2306.06308},
        pages = {arXiv:2306.06308},
          doi = {10.48550/arXiv.2306.06308},
        archivePrefix = {arXiv},
               eprint = {2306.06308},
         primaryClass = {astro-ph.CO},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230606308D},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        """,
    "desi_provabgs": """
        @article{Hahn_2023,
        title={The DESI PRObabilistic Value-added Bright Galaxy Survey (PROVABGS) Mock Challenge},
        volume={945},
        ISSN={1538-4357},
        url={http://dx.doi.org/10.3847/1538-4357/ac8983},
        DOI={10.3847/1538-4357/ac8983},
        number={1},
        journal={The Astrophysical Journal},
        publisher={American Astronomical Society},
        author={Hahn, ChangHoon and Kwon, K. J. and Tojeiro, Rita and Siudek, Malgorzata and Canning, Rebecca E. A. and Mezcua, Mar and Tinker, Jeremy L. and Brooks, David and Doel, Peter and Fanning, Kevin and Gaztañaga, Enrique and Kehoe, Robert and Landriau, Martin and Meisner, Aaron and Moustakas, John and Poppett, Claire and Tarle, Gregory and Weiner, Benjamin and Zou, Hu},
        year={2023},
        month=mar, pages={16} }

        """,
    "foundation": """
        @ARTICLE{2019ApJ...881...19J,
               author = {{Jones}, D.~O. and {Scolnic}, D.~M. and {Foley}, R.~J. and {Rest}, A. and {Kessler}, R. and {Challis}, P.~M. and {Chambers}, K.~C. and {Coulter}, D.~A. and {Dettman}, K.~G. and {Foley}, M.~M. and {Huber}, M.~E. and {Jha}, S.~W. and {Johnson}, E. and {Kilpatrick}, C.~D. and {Kirshner}, R.~P. and {Manuel}, J. and {Narayan}, G. and {Pan}, Y. -C. and {Riess}, A.~G. and {Schultz}, A.~S.~B. and {Siebert}, M.~R. and {Berger}, E. and {Chornock}, R. and {Flewelling}, H. and {Magnier}, E.~A. and {Smartt}, S.~J. and {Smith}, K.~W. and {Wainscoat}, R.~J. and {Waters}, C. and {Willman}, M.},
                title = "{The Foundation Supernova Survey: Measuring Cosmological Parameters with Supernovae from a Single Telescope}",
              journal = {\apj},
             keywords = {cosmology: observations, dark energy, supernovae: general, Astrophysics - Cosmology and Nongalactic Astrophysics},
                 year = 2019,
                month = aug,
               volume = {881},
               number = {1},
                  eid = {19},
                pages = {19},
                  doi = {10.3847/1538-4357/ab2bec},
        archivePrefix = {arXiv},
               eprint = {1811.09286},
         primaryClass = {astro-ph.CO},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJ...881...19J},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

            @ARTICLE{2018MNRAS.475..193F,
               author = {{Foley}, Ryan J. and {Scolnic}, Daniel and {Rest}, Armin and {Jha}, S.~W. and {Pan}, Y. -C. and {Riess}, A.~G. and {Challis}, P. and {Chambers}, K.~C. and {Coulter}, D.~A. and {Dettman}, K.~G. and {Foley}, M.~M. and {Fox}, O.~D. and {Huber}, M.~E. and {Jones}, D.~O. and {Kilpatrick}, C.~D. and {Kirshner}, R.~P. and {Schultz}, A.~S.~B. and {Siebert}, M.~R. and {Flewelling}, H.~A. and {Gibson}, B. and {Magnier}, E.~A. and {Miller}, J.~A. and {Primak}, N. and {Smartt}, S.~J. and {Smith}, K.~W. and {Wainscoat}, R.~J. and {Waters}, C. and {Willman}, M.},
                title = "{The Foundation Supernova Survey: motivation, design, implementation, and first data release}",
              journal = {\mnras},
             keywords = {surveys, supernovae: general, dark energy, distance scale, cosmology: observations, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Cosmology and Nongalactic Astrophysics},
                 year = 2018,
                month = mar,
               volume = {475},
               number = {1},
                pages = {193-219},
                  doi = {10.1093/mnras/stx3136},
        archivePrefix = {arXiv},
               eprint = {1711.02474},
         primaryClass = {astro-ph.HE},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2018MNRAS.475..193F},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """,
    "gaia": """
        @ARTICLE{2023A&A...674A..38G,
        author = {{Gaia Collaboration} and {Recio-Blanco}, A. and {Kordopatis}, G. and {de Laverny}, P. and {Palicio}, P.~A. and {Spagna}, A. and {Spina}, L. and {Katz}, D. and {Re Fiorentin}, P. and {Poggio}, E. and {McMillan}, P.~J. and {Vallenari}, A. and {Lattanzi}, M.~G. and {Seabroke}, G.~M. and {Casamiquela}, L. and {Bragaglia}, A. and {Antoja}, T. and {Bailer-Jones}, C.~A.~L. and {Schultheis}, M. and {Andrae}, R. and {Fouesneau}, M. and {Cropper}, M. and {Cantat-Gaudin}, T. and {Bijaoui}, A. and {Heiter}, U. and {Brown}, A.~G.~A. and {Prusti}, T. and {de Bruijne}, J.~H.~J. and {Arenou}, F. and {Babusiaux}, C. and {Biermann}, M. and {Creevey}, O.~L. and {Ducourant}, C. and {Evans}, D.~W. and {Eyer}, L. and {Guerra}, R. and {Hutton}, A. and {Jordi}, C. and {Klioner}, S.~A. and {Lammers}, U.~L. and {Lindegren}, L. and {Luri}, X. and {Mignard}, F. and {Panem}, C. and {Pourbaix}, D. and {Randich}, S. and {Sartoretti}, P. and {Soubiran}, C. and {Tanga}, P. and {Walton}, N.~A. and {Bastian}, U. and {Drimmel}, R. and {Jansen}, F. and {van Leeuwen}, F. and {Bakker}, J. and {Cacciari}, C. and {Casta{\~n}eda}, J. and {De Angeli}, F. and {Fabricius}, C. and {Fr{\'e}mat}, Y. and {Galluccio}, L. and {Guerrier}, A. and {Masana}, E. and {Messineo}, R. and {Mowlavi}, N. and {Nicolas}, C. and {Nienartowicz}, K. and {Pailler}, F. and {Panuzzo}, P. and {Riclet}, F. and {Roux}, W. and {Sordo}, R. and {Th{\'e}venin}, F. and {Gracia-Abril}, G. and {Portell}, J. and {Teyssier}, D. and {Altmann}, M. and {Audard}, M. and {Bellas-Velidis}, I. and {Benson}, K. and {Berthier}, J. and {Blomme}, R. and {Burgess}, P.~W. and {Busonero}, D. and {Busso}, G. and {C{\'a}novas}, H. and {Carry}, B. and {Cellino}, A. and {Cheek}, N. and {Clementini}, G. and {Damerdji}, Y. and {Davidson}, M. and {de Teodoro}, P. and {Nu{\~n}ez Campos}, M. and {Delchambre}, L. and {Dell'Oro}, A. and {Esquej}, P. and {Fern{\'a}ndez-Hern{\'a}ndez}, J. and {Fraile}, E. and {Garabato}, D. and {Garc{\'\i}a-Lario}, P. and {Gosset}, E. and {Haigron}, R. and {Halbwachs}, J. -L. and {Hambly}, N.~C. and {Harrison}, D.~L. and {Hern{\'a}ndez}, J. and {Hestroffer}, D. and {Hodgkin}, S.~T. and {Holl}, B. and {Jan{\ss}en}, K. and {Jevardat de Fombelle}, G. and {Jordan}, S. and {Krone-Martins}, A. and {Lanzafame}, A.~C. and {L{\"o}ffler}, W. and {Marchal}, O. and {Marrese}, P.~M. and {Moitinho}, A. and {Muinonen}, K. and {Osborne}, P. and {Pancino}, E. and {Pauwels}, T. and {Reyl{\'e}}, C. and {Riello}, M. and {Rimoldini}, L. and {Roegiers}, T. and {Rybizki}, J. and {Sarro}, L.~M. and {Siopis}, C. and {Smith}, M. and {Sozzetti}, A. and {Utrilla}, E. and {van Leeuwen}, M. and {Abbas}, U. and {{\'A}brah{\'a}m}, P. and {Abreu Aramburu}, A. and {Aerts}, C. and {Aguado}, J.~J. and {Ajaj}, M. and {Aldea-Montero}, F. and {Altavilla}, G. and {{\'A}lvarez}, M.~A. and {Alves}, J. and {Anders}, F. and {Anderson}, R.~I. and {Anglada Varela}, E. and {Baines}, D. and {Baker}, S.~G. and {Balaguer-N{\'u}{\~n}ez}, L. and {Balbinot}, E. and {Balog}, Z. and {Barache}, C. and {Barbato}, D. and {Barros}, M. and {Barstow}, M.~A. and {Bartolom{\'e}}, S. and {Bassilana}, J. -L. and {Bauchet}, N. and {Becciani}, U. and {Bellazzini}, M. and {Berihuete}, A. and {Bernet}, M. and {Bertone}, S. and {Bianchi}, L. and {Binnenfeld}, A. and {Blanco-Cuaresma}, S. and {Boch}, T. and {Bombrun}, A. and {Bossini}, D. and {Bouquillon}, S. and {Bramante}, L. and {Breedt}, E. and {Bressan}, A. and {Brouillet}, N. and {Brugaletta}, E. and {Bucciarelli}, B. and {Burlacu}, A. and {Butkevich}, A.~G. and {Buzzi}, R. and {Caffau}, E. and {Cancelliere}, R. and {Carballo}, R. and {Carlucci}, T. and {Carnerero}, M.~I. and {Carrasco}, J.~M. and {Castellani}, M. and {Castro-Ginard}, A. and {Chaoul}, L. and {Charlot}, P. and {Chemin}, L. and {Chiaramida}, V. and {Chiavassa}, A. and {Chornay}, N. and {Comoretto}, G. and {Contursi}, G. and {Cooper}, W.~J. and {Cornez}, T. and {Cowell}, S. and {Crifo}, F. and {Crosta}, M. and {Crowley}, C. and {Dafonte}, C. and {Dapergolas}, A. and {David}, P. and {De Luise}, F. and {De March}, R. and {De Ridder}, J. and {de Souza}, R. and {de Torres}, A. and {del Peloso}, E.~F. and {del Pozo}, E. and {Delbo}, M. and {Delgado}, A. and {Delisle}, J. -B. and {Demouchy}, C. and {Dharmawardena}, T.~E. and {Di Matteo}, P. and {Diakite}, S. and {Diener}, C. and {Distefano}, E. and {Dolding}, C. and {Edvardsson}, B. and {Enke}, H. and {Fabre}, C. and {Fabrizio}, M. and {Faigler}, S. and {Fedorets}, G. and {Fernique}, P. and {Figueras}, F. and {Fournier}, Y. and {Fouron}, C. and {Fragkoudi}, F. and {Gai}, M. and {Garcia-Gutierrez}, A. and {Garcia-Reinaldos}, M. and {Garc{\'\i}a-Torres}, M. and {Garofalo}, A. and {Gavel}, A. and {Gavras}, P. and {Gerlach}, E. and {Geyer}, R. and {Giacobbe}, P. and {Gilmore}, G. and {Girona}, S. and {Giuffrida}, G. and {Gomel}, R. and {Gomez}, A. and {Gonz{\'a}lez-N{\'u}{\~n}ez}, J. and {Gonz{\'a}lez-Santamar{\'\i}a}, I. and {Gonz{\'a}lez-Vidal}, J.~J. and {Granvik}, M. and {Guillout}, P. and {Guiraud}, J. and {Guti{\'e}rrez-S{\'a}nchez}, R. and {Guy}, L.~P. and {Hatzidimitriou}, D. and {Hauser}, M. and {Haywood}, M. and {Helmer}, A. and {Helmi}, A. and {Sarmiento}, M.~H. and {Hidalgo}, S.~L. and {H{\l}adczuk}, N. and {Hobbs}, D. and {Holland}, G. and {Huckle}, H.~E. and {Jardine}, K. and {Jasniewicz}, G. and {Jean-Antoine Piccolo}, A. and {Jim{\'e}nez-Arranz}, {\'O}. and {Juaristi Campillo}, J. and {Julbe}, F. and {Karbevska}, L. and {Kervella}, P. and {Khanna}, S. and {Korn}, A.~J. and {K{\'o}sp{\'a}l}, {\'A}. and {Kostrzewa-Rutkowska}, Z. and {Kruszy{\'n}ska}, K. and {Kun}, M. and {Laizeau}, P. and {Lambert}, S. and {Lanza}, A.~F. and {Lasne}, Y. and {Le Campion}, J. -F. and {Lebreton}, Y. and {Lebzelter}, T. and {Leccia}, S. and {Leclerc}, N. and {Lecoeur-Taibi}, I. and {Liao}, S. and {Licata}, E.~L. and {Lindstr{\o}m}, H.~E.~P. and {Lister}, T.~A. and {Livanou}, E. and {Lobel}, A. and {Lorca}, A. and {Loup}, C. and {Madrero Pardo}, P. and {Magdaleno Romeo}, A. and {Managau}, S. and {Mann}, R.~G. and {Manteiga}, M. and {Marchant}, J.~M. and {Marconi}, M. and {Marcos}, J. and {Marcos Santos}, M.~M.~S. and {Mar{\'\i}n Pina}, D. and {Marinoni}, S. and {Marocco}, F. and {Marshall}, D.~J. and {Martin Polo}, L. and {Mart{\'\i}n-Fleitas}, J.~M. and {Marton}, G. and {Mary}, N. and {Masip}, A. and {Massari}, D. and {Mastrobuono-Battisti}, A. and {Mazeh}, T. and {Messina}, S. and {Michalik}, D. and {Millar}, N.~R. and {Mints}, A. and {Molina}, D. and {Molinaro}, R. and {Moln{\'a}r}, L. and {Monari}, G. and {Mongui{\'o}}, M. and {Montegriffo}, P. and {Montero}, A. and {Mor}, R. and {Mora}, A. and {Morbidelli}, R. and {Morel}, T. and {Morris}, D. and {Muraveva}, T. and {Murphy}, C.~P. and {Musella}, I. and {Nagy}, Z. and {Noval}, L. and {Oca{\~n}a}, F. and {Ogden}, A. and {Ordenovic}, C. and {Osinde}, J.~O. and {Pagani}, C. and {Pagano}, I. and {Palaversa}, L. and {Pallas-Quintela}, L. and {Panahi}, A. and {Payne-Wardenaar}, S. and {Pe{\~n}alosa Esteller}, X. and {Penttil{\"a}}, A. and {Pichon}, B. and {Piersimoni}, A.~M. and {Pineau}, F. -X. and {Plachy}, E. and {Plum}, G. and {Pr{\v{s}}a}, A. and {Pulone}, L. and {Racero}, E. and {Ragaini}, S. and {Rainer}, M. and {Raiteri}, C.~M. and {Ramos}, P. and {Ramos-Lerate}, M. and {Regibo}, S. and {Richards}, P.~J. and {Rios Diaz}, C. and {Ripepi}, V. and {Riva}, A. and {Rix}, H. -W. and {Rixon}, G. and {Robichon}, N. and {Robin}, A.~C. and {Robin}, C. and {Roelens}, M. and {Rogues}, H.~R.~O. and {Rohrbasser}, L. and {Romero-G{\'o}mez}, M. and {Rowell}, N. and {Royer}, F. and {Ruz Mieres}, D. and {Rybicki}, K.~A. and {Sadowski}, G. and {S{\'a}ez N{\'u}{\~n}ez}, A. and {Sagrist{\`a} Sell{\'e}s}, A. and {Sahlmann}, J. and {Salguero}, E. and {Samaras}, N. and {Sanchez Gimenez}, V. and {Sanna}, N. and {Santove{\~n}a}, R. and {Sarasso}, M. and {Sciacca}, E. and {Segol}, M. and {Segovia}, J.~C. and {S{\'e}gransan}, D. and {Semeux}, D. and {Shahaf}, S. and {Siddiqui}, H.~I. and {Siebert}, A. and {Siltala}, L. and {Silvelo}, A. and {Slezak}, E. and {Slezak}, I. and {Smart}, R.~L. and {Snaith}, O.~N. and {Solano}, E. and {Solitro}, F. and {Souami}, D. and {Souchay}, J. and {Spoto}, F. and {Steele}, I.~A. and {Steidelm{\"u}ller}, H. and {Stephenson}, C.~A. and {S{\"u}veges}, M. and {Surdej}, J. and {Szabados}, L. and {Szegedi-Elek}, E. and {Taris}, F. and {Taylor}, M.~B. and {Teixeira}, R. and {Tolomei}, L. and {Tonello}, N. and {Torra}, F. and {Torra}, J. and {Torralba Elipe}, G. and {Trabucchi}, M. and {Tsounis}, A.~T. and {Turon}, C. and {Ulla}, A. and {Unger}, N. and {Vaillant}, M.~V. and {van Dillen}, E. and {van Reeven}, W. and {Vanel}, O. and {Vecchiato}, A. and {Viala}, Y. and {Vicente}, D. and {Voutsinas}, S. and {Weiler}, M. and {Wevers}, T. and {Wyrzykowski}, {\L}. and {Yoldas}, A. and {Yvard}, P. and {Zhao}, H. and {Zorec}, J. and {Zucker}, S. and {Zwitter}, T.},
        title = "{Gaia Data Release 3. Chemical cartography of the Milky Way}",
        journal = {\aap},
        keywords = {Galaxy: abundances, stars: abundances, Galaxy: evolution, Galaxy: kinematics and dynamics, Galaxy: disk, Galaxy: halo, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2023,
        month = jun,
        volume = {674},
          eid = {A38},
        pages = {A38},
          doi = {10.1051/0004-6361/202243511},
        archivePrefix = {arXiv},
        eprint = {2206.05534},
        primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&A...674A..38G},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }


        @ARTICLE{2016A&A...595A...1G,
        author = {{Gaia Collaboration} and {Prusti}, T. and {de Bruijne}, J.~H.~J. and {Brown}, A.~G.~A. and {Vallenari}, A. and {Babusiaux}, C. and {Bailer-Jones}, C.~A.~L. and {Bastian}, U. and {Biermann}, M. and {Evans}, D.~W. and {Eyer}, L. and {Jansen}, F. and {Jordi}, C. and {Klioner}, S.~A. and {Lammers}, U. and {Lindegren}, L. and {Luri}, X. and {Mignard}, F. and {Milligan}, D.~J. and {Panem}, C. and {Poinsignon}, V. and {Pourbaix}, D. and {Randich}, S. and {Sarri}, G. and {Sartoretti}, P. and {Siddiqui}, H.~I. and {Soubiran}, C. and {Valette}, V. and {van Leeuwen}, F. and {Walton}, N.~A. and {Aerts}, C. and {Arenou}, F. and {Cropper}, M. and {Drimmel}, R. and {H{\o}g}, E. and {Katz}, D. and {Lattanzi}, M.~G. and {O'Mullane}, W. and {Grebel}, E.~K. and {Holland}, A.~D. and {Huc}, C. and {Passot}, X. and {Bramante}, L. and {Cacciari}, C. and {Casta{\~n}eda}, J. and {Chaoul}, L. and {Cheek}, N. and {De Angeli}, F. and {Fabricius}, C. and {Guerra}, R. and {Hern{\'a}ndez}, J. and {Jean-Antoine-Piccolo}, A. and {Masana}, E. and {Messineo}, R. and {Mowlavi}, N. and {Nienartowicz}, K. and {Ord{\'o}{\~n}ez-Blanco}, D. and {Panuzzo}, P. and {Portell}, J. and {Richards}, P.~J. and {Riello}, M. and {Seabroke}, G.~M. and {Tanga}, P. and {Th{\'e}venin}, F. and {Torra}, J. and {Els}, S.~G. and {Gracia-Abril}, G. and {Comoretto}, G. and {Garcia-Reinaldos}, M. and {Lock}, T. and {Mercier}, E. and {Altmann}, M. and {Andrae}, R. and {Astraatmadja}, T.~L. and {Bellas-Velidis}, I. and {Benson}, K. and {Berthier}, J. and {Blomme}, R. and {Busso}, G. and {Carry}, B. and {Cellino}, A. and {Clementini}, G. and {Cowell}, S. and {Creevey}, O. and {Cuypers}, J. and {Davidson}, M. and {De Ridder}, J. and {de Torres}, A. and {Delchambre}, L. and {Dell'Oro}, A. and {Ducourant}, C. and {Fr{\'e}mat}, Y. and {Garc{\'\i}a-Torres}, M. and {Gosset}, E. and {Halbwachs}, J. -L. and {Hambly}, N.~C. and {Harrison}, D.~L. and {Hauser}, M. and {Hestroffer}, D. and {Hodgkin}, S.~T. and {Huckle}, H.~E. and {Hutton}, A. and {Jasniewicz}, G. and {Jordan}, S. and {Kontizas}, M. and {Korn}, A.~J. and {Lanzafame}, A.~C. and {Manteiga}, M. and {Moitinho}, A. and {Muinonen}, K. and {Osinde}, J. and {Pancino}, E. and {Pauwels}, T. and {Petit}, J. -M. and {Recio-Blanco}, A. and {Robin}, A.~C. and {Sarro}, L.~M. and {Siopis}, C. and {Smith}, M. and {Smith}, K.~W. and {Sozzetti}, A. and {Thuillot}, W. and {van Reeven}, W. and {Viala}, Y. and {Abbas}, U. and {Abreu Aramburu}, A. and {Accart}, S. and {Aguado}, J.~J. and {Allan}, P.~M. and {Allasia}, W. and {Altavilla}, G. and {{\'A}lvarez}, M.~A. and {Alves}, J. and {Anderson}, R.~I. and {Andrei}, A.~H. and {Anglada Varela}, E. and {Antiche}, E. and {Antoja}, T. and {Ant{\'o}n}, S. and {Arcay}, B. and {Atzei}, A. and {Ayache}, L. and {Bach}, N. and {Baker}, S.~G. and {Balaguer-N{\'u}{\~n}ez}, L. and {Barache}, C. and {Barata}, C. and {Barbier}, A. and {Barblan}, F. and {Baroni}, M. and {Barrado y Navascu{\'e}s}, D. and {Barros}, M. and {Barstow}, M.~A. and {Becciani}, U. and {Bellazzini}, M. and {Bellei}, G. and {Bello Garc{\'\i}a}, A. and {Belokurov}, V. and {Bendjoya}, P. and {Berihuete}, A. and {Bianchi}, L. and {Bienaym{\'e}}, O. and {Billebaud}, F. and {Blagorodnova}, N. and {Blanco-Cuaresma}, S. and {Boch}, T. and {Bombrun}, A. and {Borrachero}, R. and {Bouquillon}, S. and {Bourda}, G. and {Bouy}, H. and {Bragaglia}, A. and {Breddels}, M.~A. and {Brouillet}, N. and {Br{\"u}semeister}, T. and {Bucciarelli}, B. and {Budnik}, F. and {Burgess}, P. and {Burgon}, R. and {Burlacu}, A. and {Busonero}, D. and {Buzzi}, R. and {Caffau}, E. and {Cambras}, J. and {Campbell}, H. and {Cancelliere}, R. and {Cantat-Gaudin}, T. and {Carlucci}, T. and {Carrasco}, J.~M. and {Castellani}, M. and {Charlot}, P. and {Charnas}, J. and {Charvet}, P. and {Chassat}, F. and {Chiavassa}, A. and {Clotet}, M. and {Cocozza}, G. and {Collins}, R.~S. and {Collins}, P. and {Costigan}, G. and {Crifo}, F. and {Cross}, N.~J.~G. and {Crosta}, M. and {Crowley}, C. and {Dafonte}, C. and {Damerdji}, Y. and {Dapergolas}, A. and {David}, P. and {David}, M. and {De Cat}, P. and {de Felice}, F. and {de Laverny}, P. and {De Luise}, F. and {De March}, R. and {de Martino}, D. and {de Souza}, R. and {Debosscher}, J. and {del Pozo}, E. and {Delbo}, M. and {Delgado}, A. and {Delgado}, H.~E. and {di Marco}, F. and {Di Matteo}, P. and {Diakite}, S. and {Distefano}, E. and {Dolding}, C. and {Dos Anjos}, S. and {Drazinos}, P. and {Dur{\'a}n}, J. and {Dzigan}, Y. and {Ecale}, E. and {Edvardsson}, B. and {Enke}, H. and {Erdmann}, M. and {Escolar}, D. and {Espina}, M. and {Evans}, N.~W. and {Eynard Bontemps}, G. and {Fabre}, C. and {Fabrizio}, M. and {Faigler}, S. and {Falc{\~a}o}, A.~J. and {Farr{\`a}s Casas}, M. and {Faye}, F. and {Federici}, L. and {Fedorets}, G. and {Fern{\'a}ndez-Hern{\'a}ndez}, J. and {Fernique}, P. and {Fienga}, A. and {Figueras}, F. and {Filippi}, F. and {Findeisen}, K. and {Fonti}, A. and {Fouesneau}, M. and {Fraile}, E. and {Fraser}, M. and {Fuchs}, J. and {Furnell}, R. and {Gai}, M. and {Galleti}, S. and {Galluccio}, L. and {Garabato}, D. and {Garc{\'\i}a-Sedano}, F. and {Gar{\'e}}, P. and {Garofalo}, A. and {Garralda}, N. and {Gavras}, P. and {Gerssen}, J. and {Geyer}, R. and {Gilmore}, G. and {Girona}, S. and {Giuffrida}, G. and {Gomes}, M. and {Gonz{\'a}lez-Marcos}, A. and {Gonz{\'a}lez-N{\'u}{\~n}ez}, J. and {Gonz{\'a}lez-Vidal}, J.~J. and {Granvik}, M. and {Guerrier}, A. and {Guillout}, P. and {Guiraud}, J. and {G{\'u}rpide}, A. and {Guti{\'e}rrez-S{\'a}nchez}, R. and {Guy}, L.~P. and {Haigron}, R. and {Hatzidimitriou}, D. and {Haywood}, M. and {Heiter}, U. and {Helmi}, A. and {Hobbs}, D. and {Hofmann}, W. and {Holl}, B. and {Holland}, G. and {Hunt}, J.~A.~S. and {Hypki}, A. and {Icardi}, V. and {Irwin}, M. and {Jevardat de Fombelle}, G. and {Jofr{\'e}}, P. and {Jonker}, P.~G. and {Jorissen}, A. and {Julbe}, F. and {Karampelas}, A. and {Kochoska}, A. and {Kohley}, R. and {Kolenberg}, K. and {Kontizas}, E. and {Koposov}, S.~E. and {Kordopatis}, G. and {Koubsky}, P. and {Kowalczyk}, A. and {Krone-Martins}, A. and {Kudryashova}, M. and {Kull}, I. and {Bachchan}, R.~K. and {Lacoste-Seris}, F. and {Lanza}, A.~F. and {Lavigne}, J. -B. and {Le Poncin-Lafitte}, C. and {Lebreton}, Y. and {Lebzelter}, T. and {Leccia}, S. and {Leclerc}, N. and {Lecoeur-Taibi}, I. and {Lemaitre}, V. and {Lenhardt}, H. and {Leroux}, F. and {Liao}, S. and {Licata}, E. and {Lindstr{\o}m}, H.~E.~P. and {Lister}, T.~A. and {Livanou}, E. and {Lobel}, A. and {L{\"o}ffler}, W. and {L{\'o}pez}, M. and {Lopez-Lozano}, A. and {Lorenz}, D. and {Loureiro}, T. and {MacDonald}, I. and {Magalh{\~a}es Fernandes}, T. and {Managau}, S. and {Mann}, R.~G. and {Mantelet}, G. and {Marchal}, O. and {Marchant}, J.~M. and {Marconi}, M. and {Marie}, J. and {Marinoni}, S. and {Marrese}, P.~M. and {Marschalk{\'o}}, G. and {Marshall}, D.~J. and {Mart{\'\i}n-Fleitas}, J.~M. and {Martino}, M. and {Mary}, N. and {Matijevi{\v{c}}}, G. and {Mazeh}, T. and {McMillan}, P.~J. and {Messina}, S. and {Mestre}, A. and {Michalik}, D. and {Millar}, N.~R. and {Miranda}, B.~M.~H. and {Molina}, D. and {Molinaro}, R. and {Molinaro}, M. and {Moln{\'a}r}, L. and {Moniez}, M. and {Montegriffo}, P. and {Monteiro}, D. and {Mor}, R. and {Mora}, A. and {Morbidelli}, R. and {Morel}, T. and {Morgenthaler}, S. and {Morley}, T. and {Morris}, D. and {Mulone}, A.~F. and {Muraveva}, T. and {Musella}, I. and {Narbonne}, J. and {Nelemans}, G. and {Nicastro}, L. and {Noval}, L. and {Ord{\'e}novic}, C. and {Ordieres-Mer{\'e}}, J. and {Osborne}, P. and {Pagani}, C. and {Pagano}, I. and {Pailler}, F. and {Palacin}, H. and {Palaversa}, L. and {Parsons}, P. and {Paulsen}, T. and {Pecoraro}, M. and {Pedrosa}, R. and {Pentik{\"a}inen}, H. and {Pereira}, J. and {Pichon}, B. and {Piersimoni}, A.~M. and {Pineau}, F. -X. and {Plachy}, E. and {Plum}, G. and {Poujoulet}, E. and {Pr{\v{s}}a}, A. and {Pulone}, L. and {Ragaini}, S. and {Rago}, S. and {Rambaux}, N. and {Ramos-Lerate}, M. and {Ranalli}, P. and {Rauw}, G. and {Read}, A. and {Regibo}, S. and {Renk}, F. and {Reyl{\'e}}, C. and {Ribeiro}, R.~A. and {Rimoldini}, L. and {Ripepi}, V. and {Riva}, A. and {Rixon}, G. and {Roelens}, M. and {Romero-G{\'o}mez}, M. and {Rowell}, N. and {Royer}, F. and {Rudolph}, A. and {Ruiz-Dern}, L. and {Sadowski}, G. and {Sagrist{\`a} Sell{\'e}s}, T. and {Sahlmann}, J. and {Salgado}, J. and {Salguero}, E. and {Sarasso}, M. and {Savietto}, H. and {Schnorhk}, A. and {Schultheis}, M. and {Sciacca}, E. and {Segol}, M. and {Segovia}, J.~C. and {Segransan}, D. and {Serpell}, E. and {Shih}, I. -C. and {Smareglia}, R. and {Smart}, R.~L. and {Smith}, C. and {Solano}, E. and {Solitro}, F. and {Sordo}, R. and {Soria Nieto}, S. and {Souchay}, J. and {Spagna}, A. and {Spoto}, F. and {Stampa}, U. and {Steele}, I.~A. and {Steidelm{\"u}ller}, H. and {Stephenson}, C.~A. and {Stoev}, H. and {Suess}, F.~F. and {S{\"u}veges}, M. and {Surdej}, J. and {Szabados}, L. and {Szegedi-Elek}, E. and {Tapiador}, D. and {Taris}, F. and {Tauran}, G. and {Taylor}, M.~B. and {Teixeira}, R. and {Terrett}, D. and {Tingley}, B. and {Trager}, S.~C. and {Turon}, C. and {Ulla}, A. and {Utrilla}, E. and {Valentini}, G. and {van Elteren}, A. and {Van Hemelryck}, E. and {van Leeuwen}, M. and {Varadi}, M. and {Vecchiato}, A. and {Veljanoski}, J. and {Via}, T. and {Vicente}, D. and {Vogt}, S. and {Voss}, H. and {Votruba}, V. and {Voutsinas}, S. and {Walmsley}, G. and {Weiler}, M. and {Weingrill}, K. and {Werner}, D. and {Wevers}, T. and {Whitehead}, G. and {Wyrzykowski}, {\L}. and {Yoldas}, A. and {{\v{Z}}erjal}, M. and {Zucker}, S. and {Zurbach}, C. and {Zwitter}, T. and {Alecu}, A. and {Allen}, M. and {Allende Prieto}, C. and {Amorim}, A. and {Anglada-Escud{\'e}}, G. and {Arsenijevic}, V. and {Azaz}, S. and {Balm}, P. and {Beck}, M. and {Bernstein}, H. -H. and {Bigot}, L. and {Bijaoui}, A. and {Blasco}, C. and {Bonfigli}, M. and {Bono}, G. and {Boudreault}, S. and {Bressan}, A. and {Brown}, S. and {Brunet}, P. -M. and {Bunclark}, P. and {Buonanno}, R. and {Butkevich}, A.~G. and {Carret}, C. and {Carrion}, C. and {Chemin}, L. and {Ch{\'e}reau}, F. and {Corcione}, L. and {Darmigny}, E. and {de Boer}, K.~S. and {de Teodoro}, P. and {de Zeeuw}, P.~T. and {Delle Luche}, C. and {Domingues}, C.~D. and {Dubath}, P. and {Fodor}, F. and {Fr{\'e}zouls}, B. and {Fries}, A. and {Fustes}, D. and {Fyfe}, D. and {Gallardo}, E. and {Gallegos}, J. and {Gardiol}, D. and {Gebran}, M. and {Gomboc}, A. and {G{\'o}mez}, A. and {Grux}, E. and {Gueguen}, A. and {Heyrovsky}, A. and {Hoar}, J. and {Iannicola}, G. and {Isasi Parache}, Y. and {Janotto}, A. -M. and {Joliet}, E. and {Jonckheere}, A. and {Keil}, R. and {Kim}, D. -W. and {Klagyivik}, P. and {Klar}, J. and {Knude}, J. and {Kochukhov}, O. and {Kolka}, I. and {Kos}, J. and {Kutka}, A. and {Lainey}, V. and {LeBouquin}, D. and {Liu}, C. and {Loreggia}, D. and {Makarov}, V.~V. and {Marseille}, M.~G. and {Martayan}, C. and {Martinez-Rubi}, O. and {Massart}, B. and {Meynadier}, F. and {Mignot}, S. and {Munari}, U. and {Nguyen}, A. -T. and {Nordlander}, T. and {Ocvirk}, P. and {O'Flaherty}, K.~S. and {Olias Sanz}, A. and {Ortiz}, P. and {Osorio}, J. and {Oszkiewicz}, D. and {Ouzounis}, A. and {Palmer}, M. and {Park}, P. and {Pasquato}, E. and {Peltzer}, C. and {Peralta}, J. and {P{\'e}turaud}, F. and {Pieniluoma}, T. and {Pigozzi}, E. and {Poels}, J. and {Prat}, G. and {Prod'homme}, T. and {Raison}, F. and {Rebordao}, J.~M. and {Risquez}, D. and {Rocca-Volmerange}, B. and {Rosen}, S. and {Ruiz-Fuertes}, M.~I. and {Russo}, F. and {Sembay}, S. and {Serraller Vizcaino}, I. and {Short}, A. and {Siebert}, A. and {Silva}, H. and {Sinachopoulos}, D. and {Slezak}, E. and {Soffel}, M. and {Sosnowska}, D. and {Strai{\v{z}}ys}, V. and {ter Linden}, M. and {Terrell}, D. and {Theil}, S. and {Tiede}, C. and {Troisi}, L. and {Tsalmantza}, P. and {Tur}, D. and {Vaccari}, M. and {Vachier}, F. and {Valles}, P. and {Van Hamme}, W. and {Veltz}, L. and {Virtanen}, J. and {Wallut}, J. -M. and {Wichmann}, R. and {Wilkinson}, M.~I. and {Ziaeepour}, H. and {Zschocke}, S.},
        title = "{The Gaia mission}",
        journal = {\aap},
        keywords = {space vehicles: instruments, Galaxy: structure, astrometry, parallaxes, proper motions, telescopes, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2016,
        month = nov,
        volume = {595},
          eid = {A1},
        pages = {A1},
          doi = {10.1051/0004-6361/201629272},
        archivePrefix = {arXiv},
        eprint = {1609.04153},
        primaryClass = {astro-ph.IM},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2016A&A...595A...1G},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }


        """,
    "gz10": """

        @ARTICLE{2008MNRAS.389.1179L,
               author = {{Lintott}, Chris J. and {Schawinski}, Kevin and {Slosar}, An{\v{z}}e and {Land}, Kate and {Bamford}, Steven and {Thomas}, Daniel and {Raddick}, M. Jordan and {Nichol}, Robert C. and {Szalay}, Alex and {Andreescu}, Dan and {Murray}, Phil and {Vandenberg}, Jan},
                title = "{Galaxy Zoo: morphologies derived from visual inspection of galaxies from the Sloan Digital Sky Survey}",
              journal = {\mnras},
             keywords = {methods: data analysis, galaxies: elliptical and lenticular, cD, galaxies: general, galaxies: spiral, Astrophysics},
                 year = 2008,
                month = sep,
               volume = {389},
               number = {3},
                pages = {1179-1189},
                  doi = {10.1111/j.1365-2966.2008.13689.x},
        archivePrefix = {arXiv},
               eprint = {0804.4483},
         primaryClass = {astro-ph},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2008MNRAS.389.1179L},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        @ARTICLE{2011MNRAS.410..166L,
               author = {{Lintott}, Chris and {Schawinski}, Kevin and {Bamford}, Steven and {Slosar}, An{\r{a}}{\textthreequarters}e and {Land}, Kate and {Thomas}, Daniel and {Edmondson}, Edd and {Masters}, Karen and {Nichol}, Robert C. and {Raddick}, M. Jordan and {Szalay}, Alex and {Andreescu}, Dan and {Murray}, Phil and {Vandenberg}, Jan},
                title = "{Galaxy Zoo 1: data release of morphological classifications for nearly 900 000 galaxies}",
              journal = {\mnras},
             keywords = {methods: data analysis, galaxies: elliptical and lenticular, cD, galaxies: general, galaxies: spiral, Astrophysics - Galaxy Astrophysics, Astrophysics - Cosmology and Extragalactic Astrophysics},
                 year = 2011,
                month = jan,
               volume = {410},
               number = {1},
                pages = {166-178},
                  doi = {10.1111/j.1365-2966.2010.17432.x},
        archivePrefix = {arXiv},
               eprint = {1007.3265},
         primaryClass = {astro-ph.GA},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2011MNRAS.410..166L},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        @ARTICLE{2022MNRAS.509.3966W,
               author = {{Walmsley}, Mike and {Lintott}, Chris and {G{\'e}ron}, Tobias and {Kruk}, Sandor and {Krawczyk}, Coleman and {Willett}, Kyle W. and {Bamford}, Steven and {Kelvin}, Lee S. and {Fortson}, Lucy and {Gal}, Yarin and {Keel}, William and {Masters}, Karen L. and {Mehta}, Vihang and {Simmons}, Brooke D. and {Smethurst}, Rebecca and {Smith}, Lewis and {Baeten}, Elisabeth M. and {Macmillan}, Christine},
                title = "{Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314 000 galaxies}",
              journal = {\mnras},
             keywords = {methods: data analysis, galaxies: bar, galaxies: general, galaxies: interactions, Astrophysics - Astrophysics of Galaxies, Computer Science - Computer Vision and Pattern Recognition},
                 year = 2022,
                month = jan,
               volume = {509},
               number = {3},
                pages = {3966-3988},
                  doi = {10.1093/mnras/stab2093},
        archivePrefix = {arXiv},
               eprint = {2102.08414},
         primaryClass = {astro-ph.GA},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.3966W},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        @ARTICLE{2019AJ....157..168D,
               author = {{Dey}, Arjun and {Schlegel}, David J. and {Lang}, Dustin and {Blum}, Robert and {Burleigh}, Kaylan and {Fan}, Xiaohui and {Findlay}, Joseph R. and {Finkbeiner}, Doug and {Herrera}, David and {Juneau}, St{\'e}phanie and {Landriau}, Martin and {Levi}, Michael and {McGreer}, Ian and {Meisner}, Aaron and {Myers}, Adam D. and {Moustakas}, John and {Nugent}, Peter and {Patej}, Anna and {Schlafly}, Edward F. and {Walker}, Alistair R. and {Valdes}, Francisco and {Weaver}, Benjamin A. and {Y{\`e}che}, Christophe and {Zou}, Hu and {Zhou}, Xu and {Abareshi}, Behzad and {Abbott}, T.~M.~C. and {Abolfathi}, Bela and {Aguilera}, C. and {Alam}, Shadab and {Allen}, Lori and {Alvarez}, A. and {Annis}, James and {Ansarinejad}, Behzad and {Aubert}, Marie and {Beechert}, Jacqueline and {Bell}, Eric F. and {BenZvi}, Segev Y. and {Beutler}, Florian and {Bielby}, Richard M. and {Bolton}, Adam S. and {Brice{\~n}o}, C{\'e}sar and {Buckley-Geer}, Elizabeth J. and {Butler}, Karen and {Calamida}, Annalisa and {Carlberg}, Raymond G. and {Carter}, Paul and {Casas}, Ricard and {Castander}, Francisco J. and {Choi}, Yumi and {Comparat}, Johan and {Cukanovaite}, Elena and {Delubac}, Timoth{\'e}e and {DeVries}, Kaitlin and {Dey}, Sharmila and {Dhungana}, Govinda and {Dickinson}, Mark and {Ding}, Zhejie and {Donaldson}, John B. and {Duan}, Yutong and {Duckworth}, Christopher J. and {Eftekharzadeh}, Sarah and {Eisenstein}, Daniel J. and {Etourneau}, Thomas and {Fagrelius}, Parker A. and {Farihi}, Jay and {Fitzpatrick}, Mike and {Font-Ribera}, Andreu and {Fulmer}, Leah and {G{\"a}nsicke}, Boris T. and {Gaztanaga}, Enrique and {George}, Koshy and {Gerdes}, David W. and {Gontcho}, Satya Gontcho A. and {Gorgoni}, Claudio and {Green}, Gregory and {Guy}, Julien and {Harmer}, Diane and {Hernandez}, M. and {Honscheid}, Klaus and {Huang}, Lijuan Wendy and {James}, David J. and {Jannuzi}, Buell T. and {Jiang}, Linhua and {Joyce}, Richard and {Karcher}, Armin and {Karkar}, Sonia and {Kehoe}, Robert and {Kneib}, Jean-Paul and {Kueter-Young}, Andrea and {Lan}, Ting-Wen and {Lauer}, Tod R. and {Le Guillou}, Laurent and {Le Van Suu}, Auguste and {Lee}, Jae Hyeon and {Lesser}, Michael and {Perreault Levasseur}, Laurence and {Li}, Ting S. and {Mann}, Justin L. and {Marshall}, Robert and {Mart{\'\i}nez-V{\'a}zquez}, C.~E. and {Martini}, Paul and {du Mas des Bourboux}, H{\'e}lion and {McManus}, Sean and {Meier}, Tobias Gabriel and {M{\'e}nard}, Brice and {Metcalfe}, Nigel and {Mu{\~n}oz-Guti{\'e}rrez}, Andrea and {Najita}, Joan and {Napier}, Kevin and {Narayan}, Gautham and {Newman}, Jeffrey A. and {Nie}, Jundan and {Nord}, Brian and {Norman}, Dara J. and {Olsen}, Knut A.~G. and {Paat}, Anthony and {Palanque-Delabrouille}, Nathalie and {Peng}, Xiyan and {Poppett}, Claire L. and {Poremba}, Megan R. and {Prakash}, Abhishek and {Rabinowitz}, David and {Raichoor}, Anand and {Rezaie}, Mehdi and {Robertson}, A.~N. and {Roe}, Natalie A. and {Ross}, Ashley J. and {Ross}, Nicholas P. and {Rudnick}, Gregory and {Safonova}, Sasha and {Saha}, Abhijit and {S{\'a}nchez}, F. Javier and {Savary}, Elodie and {Schweiker}, Heidi and {Scott}, Adam and {Seo}, Hee-Jong and {Shan}, Huanyuan and {Silva}, David R. and {Slepian}, Zachary and {Soto}, Christian and {Sprayberry}, David and {Staten}, Ryan and {Stillman}, Coley M. and {Stupak}, Robert J. and {Summers}, David L. and {Sien Tie}, Suk and {Tirado}, H. and {Vargas-Maga{\~n}a}, Mariana and {Vivas}, A. Katherina and {Wechsler}, Risa H. and {Williams}, Doug and {Yang}, Jinyi and {Yang}, Qian and {Yapici}, Tolga and {Zaritsky}, Dennis and {Zenteno}, A. and {Zhang}, Kai and {Zhang}, Tianmeng and {Zhou}, Rongpu and {Zhou}, Zhimin},
                title = "{Overview of the DESI Legacy Imaging Surveys}",
              journal = {\aj},
             keywords = {catalogs, surveys, Astrophysics - Instrumentation and Methods for Astrophysics},
                 year = 2019,
                month = may,
               volume = {157},
               number = {5},
                  eid = {168},
                pages = {168},
                  doi = {10.3847/1538-3881/ab089d},
        archivePrefix = {arXiv},
               eprint = {1804.08657},
         primaryClass = {astro-ph.IM},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....157..168D},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        """,
    "hsc": """
        @article{Aihara_2022,
           title={Third data release of the Hyper Suprime-Cam Subaru Strategic Program},
           volume={74},
           ISSN={2053-051X},
           url={http://dx.doi.org/10.1093/pasj/psab122},
           DOI={10.1093/pasj/psab122},
           number={2},
           journal={Publications of the Astronomical Society of Japan},
           publisher={Oxford University Press (OUP)},
           author={Aihara, Hiroaki and AlSayyad, Yusra and Ando, Makoto and Armstrong, Robert and Bosch, James and Egami, Eiichi and Furusawa, Hisanori and Furusawa, Junko and Harasawa, Sumiko and Harikane, Yuichi and Hsieh, Bau-Ching and Ikeda, Hiroyuki and Ito, Kei and Iwata, Ikuru and Kodama, Tadayuki and Koike, Michitaro and Kokubo, Mitsuru and Komiyama, Yutaka and Li, Xiangchong and Liang, Yongming and Lin, Yen-Ting and Lupton, Robert H and Lust, Nate B and MacArthur, Lauren A and Mawatari, Ken and Mineo, Sogo and Miyatake, Hironao and Miyazaki, Satoshi and More, Surhud and Morishima, Takahiro and Murayama, Hitoshi and Nakajima, Kimihiko and Nakata, Fumiaki and Nishizawa, Atsushi J and Oguri, Masamune and Okabe, Nobuhiro and Okura, Yuki and Ono, Yoshiaki and Osato, Ken and Ouchi, Masami and Pan, Yen-Chen and Plazas Malagón, Andrés A and Price, Paul A and Reed, Sophie L and Rykoff, Eli S and Shibuya, Takatoshi and Simunovic, Mirko and Strauss, Michael A and Sugimori, Kanako and Suto, Yasushi and Suzuki, Nao and Takada, Masahiro and Takagi, Yuhei and Takata, Tadafumi and Takita, Satoshi and Tanaka, Masayuki and Tang, Shenli and Taranu, Dan S and Terai, Tsuyoshi and Toba, Yoshiki and Turner, Edwin L and Uchiyama, Hisakazu and Vijarnwannaluk, Bovornpratch and Waters, Christopher Z and Yamada, Yoshihiko and Yamamoto, Naoaki and Yamashita, Takuji},
           year={2022},
           month=feb, pages={247–272} }

        """,
    "jwst": """
        @ARTICLE{2023ApJ...947...20V, 
            author = {{Valentino}, Francesco and {Brammer}, Gabriel and {Gould}, Katriona M.~L. and {Kokorev}, Vasily and {Fujimoto}, Seiji and {Jespersen}, Christian Kragh and {Vijayan}, Aswin P. and {Weaver}, John R. and {Ito}, Kei and {Tanaka}, Masayuki and {Ilbert}, Olivier and {Magdis}, Georgios E. and {Whitaker}, Katherine E. and {Faisst}, Andreas L. and {Gallazzi}, Anna and {Gillman}, Steven and {Gim{'e}nez-Arteaga}, Clara and {G{'o}mez-Guijarro}, Carlos and {Kubo}, Mariko and {Heintz}, Kasper E. and {Hirschmann}, Michaela and {Oesch}, Pascal and {Onodera}, Masato and {Rizzo}, Francesca and {Lee}, Minju and {Strait}, Victoria and {Toft}, Sune}, 
            title = "{An Atlas of Color-selected Quiescent Galaxies at z > 3 in Public JWST Fields}", 
            journal = {pj}, 
            keywords = {Galaxy evolution, High-redshift galaxies, Galaxy quenching, Quenched galaxies, Post-starburst galaxies, Surveys, 594, 734, 2040, 2016, 2176, 1671, Astrophysics - Astrophysics of Galaxies}, 
            year = 2023, 
            month = apr, 
            volume = {947}, 
            number = {1}, 
            eid = {20}, 
            pages = {20}, 
            doi = {10.3847/1538-4357/acbefa}, 
            archivePrefix = {arXiv}, 
            eprint = {2302.10936}, 
            primaryClass = {astro-ph.GA}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...947...20V}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
          }

        @ARTICLE{2024ApJ...965L...6B, 
            author = {{Bagley}, Micaela B. and {Pirzkal}, Nor and {Finkelstein}, Steven L. and {Papovich}, Casey and {Berg}, Danielle A. and {Lotz}, Jennifer M. and {Leung}, Gene C.K. and {Ferguson}, Henry C. and {Koekemoer}, Anton M. and {Dickinson}, Mark and {Kartaltepe}, Jeyhan S. and {Kocevski}, Dale D. and {Somerville}, Rachel S. and {Yung}, L.Y. Aaron and {Backhaus}, Bren E. and {Casey}, Caitlin M. and {Castellano}, Marco and {Ch{'a}vez Ortiz}, {'O}scar A. and {Chworowsky}, Katherine and {Cox}, Isabella G. and {Dav{'e}}, Romeel and {Davis}, Kelcey and {Estrada-Carpenter}, Vicente and {Fontana}, Adriano and {Fujimoto}, Seiji and {Gardner}, Jonathan P. and {Giavalisco}, Mauro and {Grazian}, Andrea and {Grogin}, Norman A. and {Hathi}, Nimish P. and {Hutchison}, Taylor A. and {Jaskot}, Anne E. and {Jung}, Intae and {Kewley}, Lisa J. and {Kirkpatrick}, Allison and {Larson}, Rebecca L. and {Matharu}, Jasleen and {Natarajan}, Priyamvada and {Pentericci}, Laura and {P{'e}rez-Gonz{'a}lez}, Pablo G. and {Ravindranath}, Swara and {Rothberg}, Barry and {Ryan}, Russell and {Shen}, Lu and {Simons}, Raymond C. and {Snyder}, Gregory F. and {Trump}, Jonathan R. and {Wilkins}, Stephen M.}, 
            title = "{The Next Generation Deep Extragalactic Exploratory Public (NGDEEP) Survey}", 
            journal = {pjl}, 
            keywords = {Early universe, Galaxy formation, Galaxy evolution, Galaxy chemical evolution, 435, 595, 594, 580, Astrophysics - Astrophysics of Galaxies}, 
            year = 2024, 
            month = apr, 
            volume = {965}, 
            number = {1}, 
            eid = {L6}, 
            pages = {L6}, 
            doi = {10.3847/2041-8213/ad2f31}, 
            archivePrefix = {arXiv}, 
            eprint = {2302.05466}, 
            primaryClass = {astro-ph.GA}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...965L...6B},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
          }

        @ARTICLE{2023ApJ...946L..12B, 
            author = {{Bagley}, Micaela B. and {Finkelstein}, Steven L. and {Koekemoer}, Anton M. and {Ferguson}, Henry C. and {Arrabal Haro}, Pablo and {Dickinson}, Mark and {Kartaltepe}, Jeyhan S. and {Papovich}, Casey and {P{'e}rez-Gonz{'a}lez}, Pablo G. and {Pirzkal}, Nor and {Somerville}, Rachel S. and {Willmer}, Christopher N.A. and {Yang}, Guang and {Yung}, L.Y. Aaron and {Fontana}, Adriano and {Grazian}, Andrea and {Grogin}, Norman A. and {Hirschmann}, Michaela and {Kewley}, Lisa J. and {Kirkpatrick}, Allison and {Kocevski}, Dale D. and {Lotz}, Jennifer M. and {Medrano}, Aubrey and {Morales}, Alexa M. and {Pentericci}, Laura and {Ravindranath}, Swara and {Trump}, Jonathan R. and {Wilkins}, Stephen M. and {Calabr{`o}}, Antonello and {Cooper}, M.~C. and {Costantin}, Luca and {de la Vega}, Alexander and {Hilbert}, Bryan and {Hutchison}, Taylor A. and {Larson}, Rebecca L. and {Lucas}, Ray A. and {McGrath}, Elizabeth J. and {Ryan}, Russell and {Wang}, Xin and {Wuyts}, Stijn}, 
            title = "{CEERS Epoch 1 NIRCam Imaging: Reduction Methods and Simulations Enabling Early JWST Science Results}", 
            journal = {pjl}, 
            keywords = {Near infrared astronomy, Direct imaging, Astronomy data reduction, 1093, 387, 1861, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies}, 
            year = 2023, 
            month = mar, 
            volume = {946}, 
            number = {1}, 
            eid = {L12}, 
            pages = {L12}, 
            doi = {10.3847/2041-8213/acbb08}, 
            archivePrefix = {arXiv}, 
            eprint = {2211.02495}, 
            primaryClass = {astro-ph.IM}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...946L..12B}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
        }

        @ARTICLE{2023arXiv230602465E, 
              author = {{Eisenstein}, Daniel J. and {Willott}, Chris and {Alberts}, Stacey and {Arribas}, Santiago and {Bonaventura}, Nina and {Bunker}, Andrew J. and {Cameron}, Alex J. and {Carniani}, Stefano and {Charlot}, Stephane and {Curtis-Lake}, Emma and {D'Eugenio}, Francesco and {Endsley}, Ryan and {Ferruit}, Pierre and {Giardino}, Giovanna and {Hainline}, Kevin and {Hausen}, Ryan and {Jakobsen}, Peter and {Johnson}, Benjamin D. and {Maiolino}, Roberto and {Rieke}, Marcia and {Rieke}, George and {Rix}, Hans-Walter and {Robertson}, Brant and {Stark}, Daniel P. and {Tacchella}, Sandro and {Williams}, Christina C. and {Willmer}, Christopher N.A. and {Baker}, William M. and {Baum}, Stefi and {Bhatawdekar}, Rachana and {Boyett}, Kristan and {Chen}, Zuyi and {Chevallard}, Jacopo and {Circosta}, Chiara and {Curti}, Mirko and {Danhaive}, A. Lola and {DeCoursey}, Christa and {de Graaff}, Anna and {Dressler}, Alan and {Egami}, Eiichi and {Helton}, Jakob M. and {Hviding}, Raphael E. and {Ji}, Zhiyuan and {Jones}, Gareth C. and {Kumari}, Nimisha and {L{"u}tzgendorf}, Nora and {Laseter}, Isaac and {Looser}, Tobias J. and {Lyu}, Jianwei and {Maseda}, Michael V. and {Nelson}, Erica and {Parlanti}, Eleonora and {Perna}, Michele and {Pusk{'a}s}, D{'a}vid and {Rawle}, Tim and {Rodr{'\i}guez Del Pino}, Bruno and {Sandles}, Lester and {Saxena}, Aayush and {Scholtz}, Jan and {Sharpe}, Katherine and {Shivaei}, Irene and {Silcock}, Maddie S. and {Simmonds}, Charlotte and {Skarbinski}, Maya and {Smit}, Renske and {Stone}, Meredith and {Suess}, Katherine A. and {Sun}, Fengwu and {Tang}, Mengtao and {Topping}, Michael W. and {{"U}bler}, Hannah and {Villanueva}, Natalia C. and {Wallace}, Imaan E.B. and {Whitler}, Lily and {Witstok}, Joris and {Woodrum}, Charity}, 
              title = "{Overview of the JWST Advanced Deep Extragalactic Survey (JADES)}", 
              journal = {arXiv e-prints}, 
              keywords = {Astrophysics - Astrophysics of Galaxies}, 
              year = 2023, 
              month = jun, 
              eid = {arXiv:2306.02465}, 
              pages = {arXiv:2306.02465}, 
              doi = {10.48550/arXiv.2306.02465}, 
              archivePrefix = {arXiv}, 
              eprint = {2306.02465}, 
              primaryClass = {astro-ph.GA}, 
              adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230602465E}, 
              adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
          }
        """,
    "legacysurvey": """
        @ARTICLE{2019AJ....157..168D,
       author = {{Dey}, Arjun and {Schlegel}, David J. and {Lang}, Dustin and {Blum}, Robert and {Burleigh}, Kaylan and {Fan}, Xiaohui and {Findlay}, Joseph R. and {Finkbeiner}, Doug and {Herrera}, David and {Juneau}, St{\'e}phanie and {Landriau}, Martin and {Levi}, Michael and {McGreer}, Ian and {Meisner}, Aaron and {Myers}, Adam D. and {Moustakas}, John and {Nugent}, Peter and {Patej}, Anna and {Schlafly}, Edward F. and {Walker}, Alistair R. and {Valdes}, Francisco and {Weaver}, Benjamin A. and {Y{\`e}che}, Christophe and {Zou}, Hu and {Zhou}, Xu and {Abareshi}, Behzad and {Abbott}, T.~M.~C. and {Abolfathi}, Bela and {Aguilera}, C. and {Alam}, Shadab and {Allen}, Lori and {Alvarez}, A. and {Annis}, James and {Ansarinejad}, Behzad and {Aubert}, Marie and {Beechert}, Jacqueline and {Bell}, Eric F. and {BenZvi}, Segev Y. and {Beutler}, Florian and {Bielby}, Richard M. and {Bolton}, Adam S. and {Brice{\~n}o}, C{\'e}sar and {Buckley-Geer}, Elizabeth J. and {Butler}, Karen and {Calamida}, Annalisa and {Carlberg}, Raymond G. and {Carter}, Paul and {Casas}, Ricard and {Castander}, Francisco J. and {Choi}, Yumi and {Comparat}, Johan and {Cukanovaite}, Elena and {Delubac}, Timoth{\'e}e and {DeVries}, Kaitlin and {Dey}, Sharmila and {Dhungana}, Govinda and {Dickinson}, Mark and {Ding}, Zhejie and {Donaldson}, John B. and {Duan}, Yutong and {Duckworth}, Christopher J. and {Eftekharzadeh}, Sarah and {Eisenstein}, Daniel J. and {Etourneau}, Thomas and {Fagrelius}, Parker A. and {Farihi}, Jay and {Fitzpatrick}, Mike and {Font-Ribera}, Andreu and {Fulmer}, Leah and {G{\"a}nsicke}, Boris T. and {Gaztanaga}, Enrique and {George}, Koshy and {Gerdes}, David W. and {Gontcho}, Satya Gontcho A. and {Gorgoni}, Claudio and {Green}, Gregory and {Guy}, Julien and {Harmer}, Diane and {Hernandez}, M. and {Honscheid}, Klaus and {Huang}, Lijuan Wendy and {James}, David J. and {Jannuzi}, Buell T. and {Jiang}, Linhua and {Joyce}, Richard and {Karcher}, Armin and {Karkar}, Sonia and {Kehoe}, Robert and {Kneib}, Jean-Paul and {Kueter-Young}, Andrea and {Lan}, Ting-Wen and {Lauer}, Tod R. and {Le Guillou}, Laurent and {Le Van Suu}, Auguste and {Lee}, Jae Hyeon and {Lesser}, Michael and {Perreault Levasseur}, Laurence and {Li}, Ting S. and {Mann}, Justin L. and {Marshall}, Robert and {Mart{\'\i}nez-V{\'a}zquez}, C.~E. and {Martini}, Paul and {du Mas des Bourboux}, H{\'e}lion and {McManus}, Sean and {Meier}, Tobias Gabriel and {M{\'e}nard}, Brice and {Metcalfe}, Nigel and {Mu{\~n}oz-Guti{\'e}rrez}, Andrea and {Najita}, Joan and {Napier}, Kevin and {Narayan}, Gautham and {Newman}, Jeffrey A. and {Nie}, Jundan and {Nord}, Brian and {Norman}, Dara J. and {Olsen}, Knut A.~G. and {Paat}, Anthony and {Palanque-Delabrouille}, Nathalie and {Peng}, Xiyan and {Poppett}, Claire L. and {Poremba}, Megan R. and {Prakash}, Abhishek and {Rabinowitz}, David and {Raichoor}, Anand and {Rezaie}, Mehdi and {Robertson}, A.~N. and {Roe}, Natalie A. and {Ross}, Ashley J. and {Ross}, Nicholas P. and {Rudnick}, Gregory and {Safonova}, Sasha and {Saha}, Abhijit and {S{\'a}nchez}, F. Javier and {Savary}, Elodie and {Schweiker}, Heidi and {Scott}, Adam and {Seo}, Hee-Jong and {Shan}, Huanyuan and {Silva}, David R. and {Slepian}, Zachary and {Soto}, Christian and {Sprayberry}, David and {Staten}, Ryan and {Stillman}, Coley M. and {Stupak}, Robert J. and {Summers}, David L. and {Sien Tie}, Suk and {Tirado}, H. and {Vargas-Maga{\~n}a}, Mariana and {Vivas}, A. Katherina and {Wechsler}, Risa H. and {Williams}, Doug and {Yang}, Jinyi and {Yang}, Qian and {Yapici}, Tolga and {Zaritsky}, Dennis and {Zenteno}, A. and {Zhang}, Kai and {Zhang}, Tianmeng and {Zhou}, Rongpu and {Zhou}, Zhimin},
        title = "{Overview of the DESI Legacy Imaging Surveys}",
      journal = {\aj},
     keywords = {catalogs, surveys, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2019,
        month = may,
       volume = {157},
       number = {5},
          eid = {168},
        pages = {168},
          doi = {10.3847/1538-3881/ab089d},
        archivePrefix = {arXiv},
               eprint = {1804.08657},
         primaryClass = {astro-ph.IM},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....157..168D},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """,
    "plasticc": """
        @article{Kessler_2019, 
              title={Models and Simulations for the Photometric LSST Astronomical Time Series Classification Challenge (PLAsTiCC)}, 
              volume={131}, 
              ISSN={1538-3873}, 
              url={http://dx.doi.org/10.1088/1538-3873/ab26f1}, 
              DOI={10.1088/1538-3873/ab26f1}, 
              number={1003}, 
              journal={Publications of the Astronomical Society of the Pacific}, 
              publisher={IOP Publishing}, 
              author={Kessler, R. and Narayan, G. and Avelino, A. and Bachelet, E. and Biswas, R. and Brown, P. J. and Chernoff, D. F. and Connolly, A. J. and Dai, M. and Daniel, S. and Stefano, R. Di and Drout, M. R. and Galbany, L. and González-Gaitán, S. and Graham, M. L. and Hložek, R. and Ishida, E. E. O. and Guillochon, J. and Jha, S. W. and Jones, D. O. and Mandel, K. S. and Muthukrishna, D. and O’Grady, A. and Peters, C. M. and Pierel, J. R. and Ponder, K. A. and Prša, A. and Rodney, S. and Villar, V. A.}, 
              year={2019}, 
              month=jul, 
              pages={094501} 
        }

        @misc{theplasticcteam2018photometriclsstastronomicaltimeseries,
          title={The Photometric LSST Astronomical Time-series Classification Challenge (PLAsTiCC): Data set}, 
          author={The PLAsTiCC team and Tarek Allam Jr. au2 and Anita Bahmanyar and Rahul Biswas and Mi Dai and Lluís Galbany and Renée Hložek and Emille E. O. Ishida and Saurabh W. Jha and David O. Jones and Richard Kessler and Michelle Lochner and Ashish A. Mahabal and Alex I. Malz and Kaisey S. Mandel and Juan Rafael Martínez-Galarza and Jason D. McEwen and Daniel Muthukrishna and Gautham Narayan and Hiranya Peiris and Christina M. Peters and Kara Ponder and Christian N. Setzer and The LSST Dark Energy Science Collaboration and The LSST Transients and Variable Stars Science Collaboration},
          year={2018},
          eprint={1810.00001},
          archivePrefix={arXiv},
          primaryClass={astro-ph.IM},
          url={https://arxiv.org/abs/1810.00001}, 
        }
        """,
    "ps1_sne_ia": """
        @article{Scolnic_2018, 
            doi = {10.3847/1538-4357/aab9bb}, 
            url = {https://dx.doi.org/10.3847/1538-4357/aab9bb}, 
            year = {2018}, 
            month = {may}, 
            publisher = {The American Astronomical Society}, 
            volume = {859}, 
            number = {2}, 
            pages = {101},
            author = {D. M. Scolnic and D. O. Jones and A. Rest and Y. C. Pan and R. Chornock and R. J. Foley and M. E. Huber and R. Kessler and G. Narayan and A. G. Riess and S. Rodney and E. Berger and D. J. Brout and P. J. Challis and M. Drout and D. Finkbeiner and R. Lunnan and R. P. Kirshner and N. E. Sanders and E. Schlafly and S. Smartt and C. W. Stubbs and J. Tonry and W. M. Wood-Vasey and M. Foley and J. Hand and E. Johnson and W. S. Burgett and K. C. Chambers and P. W. Draper and K. W. Hodapp and N. Kaiser and R. P. Kudritzki and E. A. Magnier and N. Metcalfe and F. Bresolin and E. Gall and R. Kotak and M. McCrum and K. W. Smith}, 
            title = {The Complete Light-curve Sample of Spectroscopically Confirmed SNe Ia from Pan-STARRS1 and Cosmological Constraints from the Combined Pantheon Sample}, 
            journal = {The Astrophysical Journal}, 
            abstract = {We present optical light curves, redshifts, and classifications for  spectroscopically confirmed Type Ia supernovae (SNe Ia) discovered by the Pan-STARRS1 (PS1) Medium Deep Survey. We detail improvements to the PS1 SN photometry, astrometry, and calibration that reduce the systematic uncertainties in the PS1 SN Ia distances. We combine the subset of  PS1 SNe Ia (0.03 < z < 0.68) with useful distance estimates of SNe Ia from the Sloan Digital Sky Survey (SDSS), SNLS, and various low-z and Hubble Space Telescope samples to form the largest combined sample of SNe Ia, consisting of a total of  SNe Ia in the range of 0.01 < z < 2.3, which we call the “Pantheon Sample.” When combining Planck 2015 cosmic microwave background (CMB) measurements with the Pantheon SN sample, we find and for the wCDM model. When the SN and CMB constraints are combined with constraints from BAO and local H0 measurements, the analysis yields the most precise measurement of dark energy to date: and for the CDM model. Tension with a cosmological constant previously seen in an analysis of PS1 and low-z SNe has diminished after an increase of 2× in the statistics of the PS1 sample, improved calibration and photometry, and stricter light-curve quality cuts. We find that the systematic uncertainties in our measurements of dark energy are almost as large as the statistical uncertainties, primarily due to limitations of modeling the low-redshift sample. This must be addressed for future progress in using SNe Ia to measure dark energy.}   
          }

        @misc{chambers2019panstarrs1surveys,
              title={The Pan-STARRS1 Surveys}, 
              author={K. C. Chambers and E. A. Magnier and N. Metcalfe and H. A. Flewelling and M. E. Huber and C. Z. Waters and L. Denneau and P. W. Draper and D. Farrow and D. P. Finkbeiner and C. Holmberg and J. Koppenhoefer and P. A. Price and A. Rest and R. P. Saglia and E. F. Schlafly and S. J. Smartt and W. Sweeney and R. J. Wainscoat and W. S. Burgett and S. Chastel and T. Grav and J. N. Heasley and K. W. Hodapp and R. Jedicke and N. Kaiser and R. -P. Kudritzki and G. A. Luppino and R. H. Lupton and D. G. Monet and J. S. Morgan and P. M. Onaka and B. Shiao and C. W. Stubbs and J. L. Tonry and R. White and E. Bañados and E. F. Bell and R. Bender and E. J. Bernard and M. Boegner and F. Boffi and M. T. Botticella and A. Calamida and S. Casertano and W. -P. Chen and X. Chen and S. Cole and N. Deacon and C. Frenk and A. Fitzsimmons and S. Gezari and V. Gibbs and C. Goessl and T. Goggia and R. Gourgue and B. Goldman and P. Grant and E. K. Grebel and N. C. Hambly and G. Hasinger and A. F. Heavens and T. M. Heckman and R. Henderson and T. Henning and M. Holman and U. Hopp and W. -H. Ip and S. Isani and M. Jackson and C. D. Keyes and A. M. Koekemoer and R. Kotak and D. Le and D. Liska and K. S. Long and J. R. Lucey and M. Liu and N. F. Martin and G. Masci and B. McLean and E. Mindel and P. Misra and E. Morganson and D. N. A. Murphy and A. Obaika and G. Narayan and M. A. Nieto-Santisteban and P. Norberg and J. A. Peacock and E. A. Pier and M. Postman and N. Primak and C. Rae and A. Rai and A. Riess and A. Riffeser and H. W. Rix and S. Röser and R. Russel and L. Rutz and E. Schilbach and A. S. B. Schultz and D. Scolnic and L. Strolger and A. Szalay and S. Seitz and E. Small and K. W. Smith and D. R. Soderblom and P. Taylor and R. Thomson and A. N. Taylor and A. R. Thakar and J. Thiel and D. Thilker and D. Unger and Y. Urata and J. Valenti and J. Wagner and T. Walder and F. Walter and S. P. Watters and S. Werner and W. M. Wood-Vasey and R. Wyse},
              year={2019},
              eprint={1612.05560},
              archivePrefix={arXiv},
              primaryClass={astro-ph.IM},
              url={https://arxiv.org/abs/1612.05560}, 
        }

        @article{Magnier_2020,
        title={The Pan-STARRS Data-processing System},
        volume={251},
        ISSN={1538-4365},
        url={http://dx.doi.org/10.3847/1538-4365/abb829},
        DOI={10.3847/1538-4365/abb829},
        number={1},
        journal={The Astrophysical Journal Supplement Series},
        publisher={American Astronomical Society},
        author={Magnier, Eugene A. and Chambers, K. C. and Flewelling, H. A. and Hoblitt, J. C. and Huber, M. E. and Price, P. A. and Sweeney, W. E. and Waters, C. Z. and Denneau, L. and Draper, P. W. and Hodapp, K. W. and Jedicke, R. and Kaiser, N. and Kudritzki, R.-P. and Metcalfe, N. and Stubbs, C. W. and Wainscoat, R. J.},
        year={2020},
        month=oct, pages={3} }

        @article{Waters_2020,
           title={Pan-STARRS Pixel Processing: Detrending, Warping, Stacking},
           volume={251},
           ISSN={1538-4365},
           url={http://dx.doi.org/10.3847/1538-4365/abb82b},
           DOI={10.3847/1538-4365/abb82b},
           number={1},
           journal={The Astrophysical Journal Supplement Series},
           publisher={American Astronomical Society},
           author={Waters, C. Z. and Magnier, E. A. and Price, P. A. and Chambers, K. C. and Burgett, W. S. and Draper, P. W. and Flewelling, H. A. and Hodapp, K. W. and Huber, M. E. and Jedicke, R. and Kaiser, N. and Kudritzki, R.-P. and Lupton, R. H. and Metcalfe, N. and Rest, A. and Sweeney, W. E. and Tonry, J. L. and Wainscoat, R. J. and Wood-Vasey, W. M.},
           year={2020},
           month=oct, pages={4} }

        @article{Magnier_2020,
           title={Pan-STARRS Pixel Analysis: Source Detection and Characterization},
           volume={251},
           ISSN={1538-4365},
           url={http://dx.doi.org/10.3847/1538-4365/abb82c},
           DOI={10.3847/1538-4365/abb82c},
           number={1},
           journal={The Astrophysical Journal Supplement Series},
           publisher={American Astronomical Society},
           author={Magnier, Eugene A. and Sweeney, W. E. and Chambers, K. C. and Flewelling, H. A. and Huber, M. E. and Price, P. A. and Waters, C. Z. and Denneau, L. and Draper, P. W. and Farrow, D. and Jedicke, R. and Hodapp, K. W. and Kaiser, N. and Kudritzki, R.-P. and Metcalfe, N. and Stubbs, C. W. and Wainscoat, R. J.},
           year={2020},
           month=oct, pages={5} }

        @article{Magnier_2020,
           title={Pan-STARRS Photometric and Astrometric Calibration},
           volume={251},
           ISSN={1538-4365},
           url={http://dx.doi.org/10.3847/1538-4365/abb82a},
           DOI={10.3847/1538-4365/abb82a},
           number={1},
           journal={The Astrophysical Journal Supplement Series},
           publisher={American Astronomical Society},
           author={Magnier, Eugene. A. and Schlafly, Edward. F. and Finkbeiner, Douglas P. and Tonry, J. L. and Goldman, B. and Röser, S. and Schilbach, E. and Casertano, S. and Chambers, K. C. and Flewelling, H. A. and Huber, M. E. and Price, P. A. and Sweeney, W. E. and Waters, C. Z. and Denneau, L. and Draper, P. W. and Hodapp, K. W. and Jedicke, R. and Kaiser, N. and Kudritzki, R.-P. and Metcalfe, N. and Stubbs, C. W. and Wainscoat, R. J.},
           year={2020},
           month=oct, pages={6} }

        @article{Flewelling_2020,
           title={The Pan-STARRS1 Database and Data Products},
           volume={251},
           ISSN={1538-4365},
           url={http://dx.doi.org/10.3847/1538-4365/abb82d},
           DOI={10.3847/1538-4365/abb82d},
           number={1},
           journal={The Astrophysical Journal Supplement Series},
           publisher={American Astronomical Society},
           author={Flewelling, H. A. and Magnier, E. A. and Chambers, K. C. and Heasley, J. N. and Holmberg, C. and Huber, M. E. and Sweeney, W. and Waters, C. Z. and Calamida, A. and Casertano, S. and Chen, X. and Farrow, D. and Hasinger, G. and Henderson, R. and Long, K. S. and Metcalfe, N. and Narayan, G. and Nieto-Santisteban, M. A. and Norberg, P. and Rest, A. and Saglia, R. P. and Szalay, A. and Thakar, A. R. and Tonry, J. L. and Valenti, J. and Werner, S. and White, R. and Denneau, L. and Draper, P. W. and Hodapp, K. W. and Jedicke, R. and Kaiser, N. and Kudritzki, R. P. and Price, P. A. and Wainscoat, R. J. and Chastel, S. and McLean, B. and Postman, M. and Shiao, B.},
           year={2020},
           month=oct, pages={7} }
        """,
    "sdss": """
        @ARTICLE{2022ApJS..259...35A, 
            author = {{Abdurro'uf} and {Accetta}, Katherine and {Aerts}, Conny and {Silva Aguirre}, V{'\i}ctor and {Ahumada}, Romina and {Ajgaonkar}, Nikhil and {Filiz Ak}, N. and {Alam}, Shadab and {Allende Prieto}, Carlos and {Almeida}, Andr{'e}s and {Anders}, Friedrich and {Anderson}, Scott F. and {Andrews}, Brett H. and {Anguiano}, Borja and {Aquino-Ort{'\i}z}, Erik and {Arag{'o}n-Salamanca}, Alfonso and {Argudo-Fern{'a}ndez}, Maria and {Ata}, Metin and {Aubert}, Marie and {Avila-Reese}, Vladimir and {Badenes}, Carles and {Barb{'a}}, Rodolfo H. and {Barger}, Kat and {Barrera-Ballesteros}, Jorge K. and {Beaton}, Rachael L. and {Beers}, Timothy C. and {Belfiore}, Francesco and {Bender}, Chad F. and {Bernardi}, Mariangela and {Bershady}, Matthew A. and {Beutler}, Florian and {Bidin}, Christian Moni and {Bird}, Jonathan C. and {Bizyaev}, Dmitry and {Blanc}, Guillermo A. and {Blanton}, Michael R. and {Boardman}, Nicholas Fraser and {Bolton}, Adam S. and {Boquien}, M{'e}d{'e}ric and {Borissova}, Jura and {Bovy}, Jo and {Brandt}, W.N. and {Brown}, Jordan and {Brownstein}, Joel R. and {Brusa}, Marcella and {Buchner}, Johannes and {Bundy}, Kevin and {Burchett}, Joseph N. and {Bureau}, Martin and {Burgasser}, Adam and {Cabang}, Tuesday K. and {Campbell}, Stephanie and {Cappellari}, Michele and {Carlberg}, Joleen K. and {Wanderley}, F{'a}bio Carneiro and {Carrera}, Ricardo and {Cash}, Jennifer and {Chen}, Yan-Ping and {Chen}, Wei-Huai and {Cherinka}, Brian and {Chiappini}, Cristina and {Choi}, Peter Doohyun and {Chojnowski}, S. Drew and {Chung}, Haeun and {Clerc}, Nicolas and {Cohen}, Roger E. and {Comerford}, Julia M. and {Comparat}, Johan and {da Costa}, Luiz and {Covey}, Kevin and {Crane}, Jeffrey D. and {Cruz-Gonzalez}, Irene and {Culhane}, Connor and {Cunha}, Katia and {Dai}, Y. Sophia and {Damke}, Guillermo and {Darling}, Jeremy and {Davidson}, James W., Jr. and {Davies}, Roger and {Dawson}, Kyle and {De Lee}, Nathan and {Diamond-Stanic}, Aleksandar M. and {Cano-D{'\i}az}, Mariana and {S{'a}nchez}, Helena Dom{'\i}nguez and {Donor}, John and {Duckworth}, Chris and {Dwelly}, Tom and {Eisenstein}, Daniel J. and {Elsworth}, Yvonne P. and {Emsellem}, Eric and {Eracleous}, Mike and {Escoffier}, Stephanie and {Fan}, Xiaohui and {Farr}, Emily and {Feng}, Shuai and {Fern{'a}ndez-Trincado}, Jos{'e} G. and {Feuillet}, Diane and {Filipp}, Andreas and {Fillingham}, Sean P. and {Frinchaboy}, Peter M. and {Fromenteau}, Sebastien and {Galbany}, Llu{'\i}s and {Garc{'\i}a}, Rafael A. and {Garc{'\i}a-Hern{'a}ndez}, D.A. and {Ge}, Junqiang and {Geisler}, Doug and {Gelfand}, Joseph and {G{'e}ron}, Tobias and {Gibson}, Benjamin J. and {Goddy}, Julian and {Godoy-Rivera}, Diego and {Grabowski}, Kathleen and {Green}, Paul J. and {Greener}, Michael and {Grier}, Catherine J. and {Griffith}, Emily and {Guo}, Hong and {Guy}, Julien and {Hadjara}, Massinissa and {Harding}, Paul and {Hasselquist}, Sten and {Hayes}, Christian R. and {Hearty}, Fred and {Hern{'a}ndez}, Jes{'u}s and {Hill}, Lewis and {Hogg}, David W. and {Holtzman}, Jon A. and {Horta}, Danny and {Hsieh}, Bau-Ching and {Hsu}, Chin-Hao and {Hsu}, Yun-Hsin and {Huber}, Daniel and {Huertas-Company}, Marc and {Hutchinson}, Brian and {Hwang}, Ho Seong and {Ibarra-Medel}, H{'e}ctor J. and {Chitham}, Jacob Ider and {Ilha}, Gabriele S. and {Imig}, Julie and {Jaekle}, Will and {Jayasinghe}, Tharindu and {Ji}, Xihan and {Johnson}, Jennifer A. and {Jones}, Amy and {J{"o}nsson}, Henrik and {Katkov}, Ivan and {Khalatyan}, Arman, Dr. and {Kinemuchi}, Karen and {Kisku}, Shobhit and {Knapen}, Johan H. and {Kneib}, Jean-Paul and {Kollmeier}, Juna A. and {Kong}, Miranda and {Kounkel}, Marina and {Kreckel}, Kathryn and {Krishnarao}, Dhanesh and {Lacerna}, Ivan and {Lane}, Richard R. and {Langgin}, Rachel and {Lavender}, Ramon and {Law}, David R. and {Lazarz}, Daniel and {Leung}, Henry W. and {Leung}, Ho-Hin and {Lewis}, Hannah M. and {Li}, Cheng and {Li}, Ran and {Lian}, Jianhui and {Liang}, Fu-Heng and {Lin}, Lihwai and {Lin}, Yen-Ting and {Lin}, Sicheng and {Lintott}, Chris and {Long}, Dan and {Longa-Pe{~n}a}, Pen{'e}lope and {L{'o}pez-Cob{'a}}, Carlos and {Lu}, Shengdong and {Lundgren}, Britt F. and {Luo}, Yuanze and {Mackereth}, J. Ted and {de la Macorra}, Axel and {Mahadevan}, Suvrath and {Majewski}, Steven R. and {Manchado}, Arturo and {Mandeville}, Travis and {Maraston}, Claudia and {Margalef-Bentabol}, Berta and {Masseron}, Thomas and {Masters}, Karen L. and {Mathur}, Savita and {McDermid}, Richard M. and {Mckay}, Myles and {Merloni}, Andrea and {Merrifield}, Michael and {Meszaros}, Szabolcs and {Miglio}, Andrea and {Di Mille}, Francesco and {Minniti}, Dante and {Minsley}, Rebecca and {Monachesi}, Antonela and {Moon}, Jeongin and {Mosser}, Benoit and {Mulchaey}, John and {Muna}, Demitri and {Mu{~n}oz}, Ricardo R. and {Myers}, Adam D. and {Myers}, Natalie and {Nadathur}, Seshadri and {Nair}, Preethi and {Nandra}, Kirpal and {Neumann}, Justus and {Newman}, Jeffrey A. and {Nidever}, David L. and {Nikakhtar}, Farnik and {Nitschelm}, Christian and {O'Connell}, Julia E. and {Garma-Oehmichen}, Luis and {Luan Souza de Oliveira}, Gabriel and {Olney}, Richard and {Oravetz}, Daniel and {Ortigoza-Urdaneta}, Mario and {Osorio}, Yeisson and {Otter}, Justin and {Pace}, Zachary J. and {Padilla}, Nelson and {Pan}, Kaike and {Pan}, Hsi-An and {Parikh}, Taniya and {Parker}, James and {Peirani}, Sebastien and {Pe{~n}a Ram{'\i}rez}, Karla and {Penny}, Samantha and {Percival}, Will J. and {Perez-Fournon}, Ismael and {Pinsonneault}, Marc and {Poidevin}, Fr{'e}d{'e}rick and {Poovelil}, Vijith Jacob and {Price-Whelan}, Adrian M. and {B{'a}rbara de Andrade Queiroz}, Anna and {Raddick}, M. Jordan and {Ray}, Amy and {Rembold}, Sandro Barboza and {Riddle}, Nicole and {Riffel}, Rogemar A. and {Riffel}, Rog{'e}rio and {Rix}, Hans-Walter and {Robin}, Annie C. and {Rodr{'\i}guez-Puebla}, Aldo and {Roman-Lopes}, Alexandre and {Rom{'a}n-Z{'u}{~n}iga}, Carlos and {Rose}, Benjamin and {Ross}, Ashley J. and {Rossi}, Graziano and {Rubin}, Kate H.R. and {Salvato}, Mara and {S{'a}nchez}, Seb{'a}stian F. and {S{'a}nchez-Gallego}, Jos{'e} R. and {Sanderson}, Robyn and {Santana Rojas}, Felipe Antonio and {Sarceno}, Edgar and {Sarmiento}, Regina and {Sayres}, Conor and {Sazonova}, Elizaveta and {Schaefer}, Adam L. and {Schiavon}, Ricardo and {Schlegel}, David J. and {Schneider}, Donald P. and {Schultheis}, Mathias and {Schwope}, Axel and {Serenelli}, Aldo and {Serna}, Javier and {Shao}, Zhengyi and {Shapiro}, Griffin and {Sharma}, Anubhav and {Shen}, Yue and {Shetrone}, Matthew and {Shu}, Yiping and {Simon}, Joshua D. and {Skrutskie}, M.F. and {Smethurst}, Rebecca and {Smith}, Verne and {Sobeck}, Jennifer and {Spoo}, Taylor and {Sprague}, Dani and {Stark}, David V. and {Stassun}, Keivan G. and {Steinmetz}, Matthias and {Stello}, Dennis and {Stone-Martinez}, Alexander and {Storchi-Bergmann}, Thaisa and {Stringfellow}, Guy S. and {Stutz}, Amelia and {Su}, Yung-Chau and {Taghizadeh-Popp}, Manuchehr and {Talbot}, Michael S. and {Tayar}, Jamie and {Telles}, Eduardo and {Teske}, Johanna and {Thakar}, Ani and {Theissen}, Christopher and {Tkachenko}, Andrew and {Thomas}, Daniel and {Tojeiro}, Rita and {Hernandez Toledo}, Hector and {Troup}, Nicholas W. and {Trump}, Jonathan R. and {Trussler}, James and {Turner}, Jacqueline and {Tuttle}, Sarah and {Unda-Sanzana}, Eduardo and {V{'a}zquez-Mata}, Jos{'e} Antonio and {Valentini}, Marica and {Valenzuela}, Octavio and {Vargas-Gonz{'a}lez}, Jaime and {Vargas-Maga{~n}a}, Mariana and {Alfaro}, Pablo Vera and {Villanova}, Sandro and {Vincenzo}, Fiorenzo and {Wake}, David and {Warfield}, Jack T. and {Washington}, Jessica Diane and {Weaver}, Benjamin Alan and {Weijmans}, Anne-Marie and {Weinberg}, David H. and {Weiss}, Achim and {Westfall}, Kyle B. and {Wild}, Vivienne and {Wilde}, Matthew C. and {Wilson}, John C. and {Wilson}, Robert F. and {Wilson}, Mikayla and {Wolf}, Julien and {Wood-Vasey}, W.~M. and {Yan}, Renbin and {Zamora}, Olga and {Zasowski}, Gail and {Zhang}, Kai and {Zhao}, Cheng and {Zheng}, Zheng and {Zheng}, Zheng and {Zhu}, Kai}, 
            title = "{The Seventeenth Data Release of the Sloan Digital Sky Surveys: Complete Release of MaNGA, MaStar, and APOGEE-2 Data}", 
            journal = {pjs}, 
            keywords = {Astronomy data acquisition, Astronomy databases, Surveys, 1860, 83, 1671, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics}, 
            year = 2022, 
            month = apr, 
            volume = {259}, 
            number = {2}, 
            eid = {35}, 
            pages = {35}, 
            doi = {10.3847/1538-4365/ac4414}, 
            archivePrefix = {arXiv}, 
            eprint = {2112.02026}, 
            primaryClass = {astro-ph.GA}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
          }

                @ARTICLE{2017AJ....154...28B,
           author = {{Blanton}, Michael R. and {Bershady}, Matthew A. and {Abolfathi}, Bela and {Albareti}, Franco D. and {Allende Prieto}, Carlos and {Almeida}, Andres and {Alonso-Garc{\'\i}a}, Javier and {Anders}, Friedrich and {Anderson}, Scott F. and {Andrews}, Brett and {Aquino-Ort{\'\i}z}, Erik and {Arag{\'o}n-Salamanca}, Alfonso and {Argudo-Fern{\'a}ndez}, Maria and {Armengaud}, Eric and {Aubourg}, Eric and {Avila-Reese}, Vladimir and {Badenes}, Carles and {Bailey}, Stephen and {Barger}, Kathleen A. and {Barrera-Ballesteros}, Jorge and {Bartosz}, Curtis and {Bates}, Dominic and {Baumgarten}, Falk and {Bautista}, Julian and {Beaton}, Rachael and {Beers}, Timothy C. and {Belfiore}, Francesco and {Bender}, Chad F. and {Berlind}, Andreas A. and {Bernardi}, Mariangela and {Beutler}, Florian and {Bird}, Jonathan C. and {Bizyaev}, Dmitry and {Blanc}, Guillermo A. and {Blomqvist}, Michael and {Bolton}, Adam S. and {Boquien}, M{\'e}d{\'e}ric and {Borissova}, Jura and {van den Bosch}, Remco and {Bovy}, Jo and {Brandt}, William N. and {Brinkmann}, Jonathan and {Brownstein}, Joel R. and {Bundy}, Kevin and {Burgasser}, Adam J. and {Burtin}, Etienne and {Busca}, Nicol{\'a}s G. and {Cappellari}, Michele and {Delgado Carigi}, Maria Leticia and {Carlberg}, Joleen K. and {Carnero Rosell}, Aurelio and {Carrera}, Ricardo and {Chanover}, Nancy J. and {Cherinka}, Brian and {Cheung}, Edmond and {G{\'o}mez Maqueo Chew}, Yilen and {Chiappini}, Cristina and {Choi}, Peter Doohyun and {Chojnowski}, Drew and {Chuang}, Chia-Hsun and {Chung}, Haeun and {Cirolini}, Rafael Fernando and {Clerc}, Nicolas and {Cohen}, Roger E. and {Comparat}, Johan and {da Costa}, Luiz and {Cousinou}, Marie-Claude and {Covey}, Kevin and {Crane}, Jeffrey D. and {Croft}, Rupert A.~C. and {Cruz-Gonzalez}, Irene and {Garrido Cuadra}, Daniel and {Cunha}, Katia and {Damke}, Guillermo J. and {Darling}, Jeremy and {Davies}, Roger and {Dawson}, Kyle and {de la Macorra}, Axel and {Dell'Agli}, Flavia and {De Lee}, Nathan and {Delubac}, Timoth{\'e}e and {Di Mille}, Francesco and {Diamond-Stanic}, Aleks and {Cano-D{\'\i}az}, Mariana and {Donor}, John and {Downes}, Juan Jos{\'e} and {Drory}, Niv and {du Mas des Bourboux}, H{\'e}lion and {Duckworth}, Christopher J. and {Dwelly}, Tom and {Dyer}, Jamie and {Ebelke}, Garrett and {Eigenbrot}, Arthur D. and {Eisenstein}, Daniel J. and {Emsellem}, Eric and {Eracleous}, Mike and {Escoffier}, Stephanie and {Evans}, Michael L. and {Fan}, Xiaohui and {Fern{\'a}ndez-Alvar}, Emma and {Fernandez-Trincado}, J.~G. and {Feuillet}, Diane K. and {Finoguenov}, Alexis and {Fleming}, Scott W. and {Font-Ribera}, Andreu and {Fredrickson}, Alexander and {Freischlad}, Gordon and {Frinchaboy}, Peter M. and {Fuentes}, Carla E. and {Galbany}, Llu{\'\i}s and {Garcia-Dias}, R. and {Garc{\'\i}a-Hern{\'a}ndez}, D.~A. and {Gaulme}, Patrick and {Geisler}, Doug and {Gelfand}, Joseph D. and {Gil-Mar{\'\i}n}, H{\'e}ctor and {Gillespie}, Bruce A. and {Goddard}, Daniel and {Gonzalez-Perez}, Violeta and {Grabowski}, Kathleen and {Green}, Paul J. and {Grier}, Catherine J. and {Gunn}, James E. and {Guo}, Hong and {Guy}, Julien and {Hagen}, Alex and {Hahn}, ChangHoon and {Hall}, Matthew and {Harding}, Paul and {Hasselquist}, Sten and {Hawley}, Suzanne L. and {Hearty}, Fred and {Gonzalez Hern{\'a}ndez}, Jonay I. and {Ho}, Shirley and {Hogg}, David W. and {Holley-Bockelmann}, Kelly and {Holtzman}, Jon A. and {Holzer}, Parker H. and {Huehnerhoff}, Joseph and {Hutchinson}, Timothy A. and {Hwang}, Ho Seong and {Ibarra-Medel}, H{\'e}ctor J. and {da Silva Ilha}, Gabriele and {Ivans}, Inese I. and {Ivory}, KeShawn and {Jackson}, Kelly and {Jensen}, Trey W. and {Johnson}, Jennifer A. and {Jones}, Amy and {J{\"o}nsson}, Henrik and {Jullo}, Eric and {Kamble}, Vikrant and {Kinemuchi}, Karen and {Kirkby}, David and {Kitaura}, Francisco-Shu and {Klaene}, Mark and {Knapp}, Gillian R. and {Kneib}, Jean-Paul and {Kollmeier}, Juna A. and {Lacerna}, Ivan and {Lane}, Richard R. and {Lang}, Dustin and {Law}, David R. and {Lazarz}, Daniel and {Lee}, Youngbae and {Le Goff}, Jean-Marc and {Liang}, Fu-Heng and {Li}, Cheng and {Li}, Hongyu and {Lian}, Jianhui and {Lima}, Marcos and {Lin}, Lihwai and {Lin}, Yen-Ting and {Bertran de Lis}, Sara and {Liu}, Chao and {de Icaza Lizaola}, Miguel Angel C. and {Long}, Dan and {Lucatello}, Sara and {Lundgren}, Britt and {MacDonald}, Nicholas K. and {Deconto Machado}, Alice and {MacLeod}, Chelsea L. and {Mahadevan}, Suvrath and {Geimba Maia}, Marcio Antonio and {Maiolino}, Roberto and {Majewski}, Steven R. and {Malanushenko}, Elena and {Malanushenko}, Viktor and {Manchado}, Arturo and {Mao}, Shude and {Maraston}, Claudia and {Marques-Chaves}, Rui and {Masseron}, Thomas and {Masters}, Karen L. and {McBride}, Cameron K. and {McDermid}, Richard M. and {McGrath}, Brianne and {McGreer}, Ian D. and {Medina Pe{\~n}a}, Nicol{\'a}s and {Melendez}, Matthew and {Merloni}, Andrea and {Merrifield}, Michael R. and {Meszaros}, Szabolcs and {Meza}, Andres and {Minchev}, Ivan and {Minniti}, Dante and {Miyaji}, Takamitsu and {More}, Surhud and {Mulchaey}, John and {M{\"u}ller-S{\'a}nchez}, Francisco and {Muna}, Demitri and {Munoz}, Ricardo R. and {Myers}, Adam D. and {Nair}, Preethi and {Nandra}, Kirpal and {Correa do Nascimento}, Janaina and {Negrete}, Alenka and {Ness}, Melissa and {Newman}, Jeffrey A. and {Nichol}, Robert C. and {Nidever}, David L. and {Nitschelm}, Christian and {Ntelis}, Pierros and {O'Connell}, Julia E. and {Oelkers}, Ryan J. and {Oravetz}, Audrey and {Oravetz}, Daniel and {Pace}, Zach and {Padilla}, Nelson and {Palanque-Delabrouille}, Nathalie and {Alonso Palicio}, Pedro and {Pan}, Kaike and {Parejko}, John K. and {Parikh}, Taniya and {P{\^a}ris}, Isabelle and {Park}, Changbom and {Patten}, Alim Y. and {Peirani}, Sebastien and {Pellejero-Ibanez}, Marcos and {Penny}, Samantha and {Percival}, Will J. and {Perez-Fournon}, Ismael and {Petitjean}, Patrick and {Pieri}, Matthew M. and {Pinsonneault}, Marc and {Pisani}, Alice and {Poleski}, Rados{\l}aw and {Prada}, Francisco and {Prakash}, Abhishek and {Queiroz}, Anna B{\'a}rbara de Andrade and {Raddick}, M. Jordan and {Raichoor}, Anand and {Barboza Rembold}, Sandro and {Richstein}, Hannah and {Riffel}, Rogemar A. and {Riffel}, Rog{\'e}rio and {Rix}, Hans-Walter and {Robin}, Annie C. and {Rockosi}, Constance M. and {Rodr{\'\i}guez-Torres}, Sergio and {Roman-Lopes}, A. and {Rom{\'a}n-Z{\'u}{\~n}iga}, Carlos and {Rosado}, Margarita and {Ross}, Ashley J. and {Rossi}, Graziano and {Ruan}, John and {Ruggeri}, Rossana and {Rykoff}, Eli S. and {Salazar-Albornoz}, Salvador and {Salvato}, Mara and {S{\'a}nchez}, Ariel G. and {Aguado}, D.~S. and {S{\'a}nchez-Gallego}, Jos{\'e} R. and {Santana}, Felipe A. and {Santiago}, Bas{\'\i}lio Xavier and {Sayres}, Conor and {Schiavon}, Ricardo P. and {da Silva Schimoia}, Jaderson and {Schlafly}, Edward F. and {Schlegel}, David J. and {Schneider}, Donald P. and {Schultheis}, Mathias and {Schuster}, William J. and {Schwope}, Axel and {Seo}, Hee-Jong and {Shao}, Zhengyi and {Shen}, Shiyin and {Shetrone}, Matthew and {Shull}, Michael and {Simon}, Joshua D. and {Skinner}, Danielle and {Skrutskie}, M.~F. and {Slosar}, An{\v{z}}e and {Smith}, Verne V. and {Sobeck}, Jennifer S. and {Sobreira}, Flavia and {Somers}, Garrett and {Souto}, Diogo and {Stark}, David V. and {Stassun}, Keivan and {Stauffer}, Fritz and {Steinmetz}, Matthias and {Storchi-Bergmann}, Thaisa and {Streblyanska}, Alina and {Stringfellow}, Guy S. and {Su{\'a}rez}, Genaro and {Sun}, Jing and {Suzuki}, Nao and {Szigeti}, Laszlo and {Taghizadeh-Popp}, Manuchehr and {Tang}, Baitian and {Tao}, Charling and {Tayar}, Jamie and {Tembe}, Mita and {Teske}, Johanna and {Thakar}, Aniruddha R. and {Thomas}, Daniel and {Thompson}, Benjamin A. and {Tinker}, Jeremy L. and {Tissera}, Patricia and {Tojeiro}, Rita and {Hernandez Toledo}, Hector and {de la Torre}, Sylvain and {Tremonti}, Christy and {Troup}, Nicholas W. and {Valenzuela}, Octavio and {Martinez Valpuesta}, Inma and {Vargas-Gonz{\'a}lez}, Jaime and {Vargas-Maga{\~n}a}, Mariana and {Vazquez}, Jose Alberto and {Villanova}, Sandro and {Vivek}, M. and {Vogt}, Nicole and {Wake}, David and {Walterbos}, Rene and {Wang}, Yuting and {Weaver}, Benjamin Alan and {Weijmans}, Anne-Marie and {Weinberg}, David H. and {Westfall}, Kyle B. and {Whelan}, David G. and {Wild}, Vivienne and {Wilson}, John and {Wood-Vasey}, W.~M. and {Wylezalek}, Dominika and {Xiao}, Ting and {Yan}, Renbin and {Yang}, Meng and {Ybarra}, Jason E. and {Y{\`e}che}, Christophe and {Zakamska}, Nadia and {Zamora}, Olga and {Zarrouk}, Pauline and {Zasowski}, Gail and {Zhang}, Kai and {Zhao}, Gong-Bo and {Zheng}, Zheng and {Zheng}, Zheng and {Zhou}, Xu and {Zhou}, Zhi-Min and {Zhu}, Guangtun B. and {Zoccali}, Manuela and {Zou}, Hu},
            title = "{Sloan Digital Sky Survey IV: Mapping the Milky Way, Nearby Galaxies, and the Distant Universe}",
              journal = {\aj},
             keywords = {cosmology: observations, galaxies: general, Galaxy: general, instrumentation: spectrographs, stars: general, surveys, Astrophysics - Astrophysics of Galaxies},
                 year = 2017,
                month = jul,
               volume = {154},
               number = {1},
                  eid = {28},
                pages = {28},
                  doi = {10.3847/1538-3881/aa7567},
            archivePrefix = {arXiv},
                   eprint = {1703.00052},
             primaryClass = {astro-ph.GA},
                   adsurl = {https://ui.adsabs.harvard.edu/abs/2017AJ....154...28B},
                  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }

        @ARTICLE{2022ApJS..259...35A, 
            author = {{Abdurro'uf} and et al.}, 
            title = "{The Seventeenth Data Release of the Sloan Digital Sky Surveys: Complete Release of MaNGA, MaStar, and APOGEE-2 Data}", 
            journal = {pjs}, 
            keywords = {Astronomy data acquisition, Astronomy databases, Surveys, 1860, 83, 1671, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics}, 
            year = 2022, 
            month = apr, 
            volume = {259}, 
            number = {2}, 
            eid = {35}, 
            pages = {35}, 
            doi = {10.3847/1538-4365/ac4414}, 
            archivePrefix = {arXiv}, 
            eprint = {2112.02026}, 
            primaryClass = {astro-ph.GA}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}   
          }

        @ARTICLE{2017AJ....154...94M,
           author = {{Majewski}, S.~R. and {Schiavon}, R.~P. and {Frinchaboy}, P.~M. and 
            {Allende Prieto}, C. and {Barkhouser}, R. and {Bizyaev}, D. and 
            {Blank}, B. and {Brunner}, S. and {Burton}, A. and {Carrera}, R. and 
            {Chojnowski}, S.~D. and {Cunha}, K. and {Epstein}, C. and {Fitzgerald}, G. and 
            {Garc{\'{\i}}a P{\'e}rez}, A.~E. and {Hearty}, F.~R. and {Henderson}, C. and 
            {Holtzman}, J.~A. and {Johnson}, J.~A. and {Lam}, C.~R. and 
            {Lawler}, J.~E. and {Maseman}, P. and {M{\'e}sz{\'a}ros}, S. and 
            {Nelson}, M. and {Nguyen}, D.~C. and {Nidever}, D.~L. and {Pinsonneault}, M. and 
            {Shetrone}, M. and {Smee}, S. and {Smith}, V.~V. and {Stolberg}, T. and 
            {Skrutskie}, M.~F. and {Walker}, E. and {Wilson}, J.~C. and 
            {Zasowski}, G. and {Anders}, F. and {Basu}, S. and {Beland}, S. and 
            {Blanton}, M.~R. and {Bovy}, J. and {Brownstein}, J.~R. and 
            {Carlberg}, J. and {Chaplin}, W. and {Chiappini}, C. and {Eisenstein}, D.~J. and 
            {Elsworth}, Y. and {Feuillet}, D. and {Fleming}, S.~W. and {Galbraith-Frew}, J. and 
            {Garc{\'{\i}}a}, R.~A. and {Garc{\'{\i}}a-Hern{\'a}ndez}, D.~A. and 
            {Gillespie}, B.~A. and {Girardi}, L. and {Gunn}, J.~E. and {Hasselquist}, S. and 
            {Hayden}, M.~R. and {Hekker}, S. and {Ivans}, I. and {Kinemuchi}, K. and 
            {Klaene}, M. and {Mahadevan}, S. and {Mathur}, S. and {Mosser}, B. and 
            {Muna}, D. and {Munn}, J.~A. and {Nichol}, R.~C. and {O'Connell}, R.~W. and 
            {Parejko}, J.~K. and {Robin}, A.~C. and {Rocha-Pinto}, H. and 
            {Schultheis}, M. and {Serenelli}, A.~M. and {Shane}, N. and 
            {Silva Aguirre}, V. and {Sobeck}, J.~S. and {Thompson}, B. and 
            {Troup}, N.~W. and {Weinberg}, D.~H. and {Zamora}, O.},
            title = "{The Apache Point Observatory Galactic Evolution Experiment (APOGEE)}",
            journal = {\aj},
            archivePrefix = "arXiv",
            eprint = {1509.05420},
            primaryClass = "astro-ph.IM",
            keywords = {Galaxy: abundances, Galaxy: evolution, Galaxy: formation, Galaxy: kinematics and dynamics, Galaxy: stellar content, Galaxy: structure},
            year = 2017,
            month = sep,
            volume = 154,
              eid = {94},
            pages = {94},
              doi = {10.3847/1538-3881/aa784d},
            adsurl = {http://adsabs.harvard.edu/abs/2017AJ....154...94M},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }

        @ARTICLE{2019PASP..131e5001W,
               author = {{Wilson}, J.~C. and {Hearty}, F.~R. and {Skrutskie}, M.~F. and {Majewski}, S.~R. and {Holtzman}, J.~A. and {Eisenstein}, D. and {Gunn}, J. and {Blank}, B. and {Henderson}, C. and {Smee}, S. and {Nelson}, M. and {Nidever}, D. and {Arns}, J. and {Barkhouser}, R. and {Barr}, J. and {Beland}, S. and {Bershady}, M.~A. and {Blanton}, M.~R. and {Brunner}, S. and {Burton}, A. and {Carey}, L. and {Carr}, M. and {Colque}, J.~P. and {Crane}, J. and {Damke}, G.~J. and {Davidson}, J.~W., Jr. and {Dean}, J. and {Di Mille}, F. and {Don}, K.~W. and {Ebelke}, G. and {Evans}, M. and {Fitzgerald}, G. and {Gillespie}, B. and {Hall}, M. and {Harding}, A. and {Harding}, P. and {Hammond}, R. and {Hancock}, D. and {Harrison}, C. and {Hope}, S. and {Horne}, T. and {Karakla}, J. and {Lam}, C. and {Leger}, F. and {MacDonald}, N. and {Maseman}, P. and {Matsunari}, J. and {Melton}, S. and {Mitcheltree}, T. and {O'Brien}, T. and {O'Connell}, R.~W. and {Patten}, A. and {Richardson}, W. and {Rieke}, G. and {Rieke}, M. and {Roman-Lopes}, A. and {Schiavon}, R.~P. and {Sobeck}, J.~S. and {Stolberg}, T. and {Stoll}, R. and {Tembe}, M. and {Trujillo}, J.~D. and {Uomoto}, A. and {Vernieri}, M. and {Walker}, E. and {Weinberg}, D.~H. and {Young}, E. and {Anthony-Brumfield}, B. and {Bizyaev}, D. and {Breslauer}, B. and {De Lee}, N. and {Downey}, J. and {Halverson}, S. and {Huehnerhoff}, J. and {Klaene}, M. and {Leon}, E. and {Long}, D. and {Mahadevan}, S. and {Malanushenko}, E. and {Nguyen}, D.~C. and {Owen}, R. and {S{\'a}nchez-Gallego}, J.~R. and {Sayres}, C. and {Shane}, N. and {Shectman}, S.~A. and {Shetrone}, M. and {Skinner}, D. and {Stauffer}, F. and {Zhao}, B.},
                title = "{The Apache Point Observatory Galactic Evolution Experiment (APOGEE) Spectrographs}",
              journal = {\pasp},
             keywords = {Astrophysics - Instrumentation and Methods for Astrophysics},
                 year = 2019,
                month = may,
               volume = {131},
               number = {999},
                pages = {055001},
                  doi = {10.1088/1538-3873/ab0075},
        archivePrefix = {arXiv},
               eprint = {1902.00928},
         primaryClass = {astro-ph.IM},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2019PASP..131e5001W},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        @ARTICLE{2016AJ....151..144G, 
            author = {{Garc{'\i}a P{'e}rez}, Ana E. and {Allende Prieto}, Carlos and {Holtzman}, Jon A. and {Shetrone}, Matthew and {M{'e}sz{'a}ros}, Szabolcs and {Bizyaev}, Dmitry and {Carrera}, Ricardo and {Cunha}, Katia and {Garc{'\i}a-Hern{'a}ndez}, D.~A. and {Johnson}, Jennifer A. and {Majewski}, Steven R. and {Nidever}, David L. and {Schiavon}, Ricardo P. and {Shane}, Neville and {Smith}, Verne V. and {Sobeck}, Jennifer and {Troup}, Nicholas and {Zamora}, Olga and {Weinberg}, David H. and {Bovy}, Jo and {Eisenstein}, Daniel J. and {Feuillet}, Diane and {Frinchaboy}, Peter M. and {Hayden}, Michael R. and {Hearty}, Fred R. and {Nguyen}, Duy C. and {O'Connell}, Robert W. and {Pinsonneault}, Marc H. and {Wilson}, John C. and {Zasowski}, Gail}, 
            title = "{ASPCAP: The APOGEE Stellar Parameter and Chemical Abundances Pipeline}", 
            journal = {j}, 
            keywords = {Galaxy: center, Galaxy: structure, methods: data analysis, stars: abundances, stars: atmospheres, Astrophysics - Solar and Stellar Astrophysics}, 
            year = 2016, 
            month = jun, 
            volume = {151}, 
            number = {6}, 
            eid = {144}, 
            pages = {144}, 
            doi = {10.3847/0004-6256/151/6/144}, 
            archivePrefix = {arXiv}, 
            eprint = {1510.07635}, 
            primaryClass = {astro-ph.SR}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2016AJ....151..144G}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
          }
        """,
    "snls": """
        @ARTICLE{2010A&A...523A...7G, 
            author = {{Guy}, J. and {Sullivan}, M. and {Conley}, A. and {Regnault}, N. and {Astier}, P. and {Balland}, C. and {Basa}, S. and {Carlberg}, R.G. and {Fouchez}, D. and {Hardin}, D. and {Hook}, I.M. and {Howell}, D.A. and {Pain}, R. and {Palanque-Delabrouille}, N. and {Perrett}, K.M. and {Pritchet}, C.J. and {Rich}, J. and {Ruhlmann-Kleider}, V. and {Balam}, D. and {Baumont}, S. and {Ellis}, R.S. and {Fabbro}, S. and {Fakhouri}, H.K. and {Fourmanoit}, N. and {Gonz{'a}lez-Gait{'a}n}, S. and {Graham}, M.L. and {Hsiao}, E. and {Kronborg}, T. and {Lidman}, C. and {Mourao}, A.M. and {Perlmutter}, S. and {Ripoche}, P. and {Suzuki}, N. and {Walker}, E.S.},
            title = "{The Supernova Legacy Survey 3-year sample: Type Ia supernovae photometric distances and cosmological constraints}", 
            journal = {ap}, 
            keywords = {supernovae: general, cosmology: observations, Astrophysics - Cosmology and Nongalactic Astrophysics}, 
            year = 2010, 
            month = nov, 
            volume = {523}, 
            eid = {A7}, 
            pages = {A7}, 
            doi = {10.1051/0004-6361/201014468}, 
            archivePrefix = {arXiv}, 
            eprint = {1010.4743}, 
            primaryClass = {astro-ph.CO}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2010A&A...523A...7G}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
        }
        """,
    "ssl_legacysurvey": """
        @ARTICLE{2019AJ....157..168D,
       author = {{Dey}, Arjun and {Schlegel}, David J. and {Lang}, Dustin and {Blum}, Robert and {Burleigh}, Kaylan and {Fan}, Xiaohui and {Findlay}, Joseph R. and {Finkbeiner}, Doug and {Herrera}, David and {Juneau}, St{\'e}phanie and {Landriau}, Martin and {Levi}, Michael and {McGreer}, Ian and {Meisner}, Aaron and {Myers}, Adam D. and {Moustakas}, John and {Nugent}, Peter and {Patej}, Anna and {Schlafly}, Edward F. and {Walker}, Alistair R. and {Valdes}, Francisco and {Weaver}, Benjamin A. and {Y{\`e}che}, Christophe and {Zou}, Hu and {Zhou}, Xu and {Abareshi}, Behzad and {Abbott}, T.~M.~C. and {Abolfathi}, Bela and {Aguilera}, C. and {Alam}, Shadab and {Allen}, Lori and {Alvarez}, A. and {Annis}, James and {Ansarinejad}, Behzad and {Aubert}, Marie and {Beechert}, Jacqueline and {Bell}, Eric F. and {BenZvi}, Segev Y. and {Beutler}, Florian and {Bielby}, Richard M. and {Bolton}, Adam S. and {Brice{\~n}o}, C{\'e}sar and {Buckley-Geer}, Elizabeth J. and {Butler}, Karen and {Calamida}, Annalisa and {Carlberg}, Raymond G. and {Carter}, Paul and {Casas}, Ricard and {Castander}, Francisco J. and {Choi}, Yumi and {Comparat}, Johan and {Cukanovaite}, Elena and {Delubac}, Timoth{\'e}e and {DeVries}, Kaitlin and {Dey}, Sharmila and {Dhungana}, Govinda and {Dickinson}, Mark and {Ding}, Zhejie and {Donaldson}, John B. and {Duan}, Yutong and {Duckworth}, Christopher J. and {Eftekharzadeh}, Sarah and {Eisenstein}, Daniel J. and {Etourneau}, Thomas and {Fagrelius}, Parker A. and {Farihi}, Jay and {Fitzpatrick}, Mike and {Font-Ribera}, Andreu and {Fulmer}, Leah and {G{\"a}nsicke}, Boris T. and {Gaztanaga}, Enrique and {George}, Koshy and {Gerdes}, David W. and {Gontcho}, Satya Gontcho A. and {Gorgoni}, Claudio and {Green}, Gregory and {Guy}, Julien and {Harmer}, Diane and {Hernandez}, M. and {Honscheid}, Klaus and {Huang}, Lijuan Wendy and {James}, David J. and {Jannuzi}, Buell T. and {Jiang}, Linhua and {Joyce}, Richard and {Karcher}, Armin and {Karkar}, Sonia and {Kehoe}, Robert and {Kneib}, Jean-Paul and {Kueter-Young}, Andrea and {Lan}, Ting-Wen and {Lauer}, Tod R. and {Le Guillou}, Laurent and {Le Van Suu}, Auguste and {Lee}, Jae Hyeon and {Lesser}, Michael and {Perreault Levasseur}, Laurence and {Li}, Ting S. and {Mann}, Justin L. and {Marshall}, Robert and {Mart{\'\i}nez-V{\'a}zquez}, C.~E. and {Martini}, Paul and {du Mas des Bourboux}, H{\'e}lion and {McManus}, Sean and {Meier}, Tobias Gabriel and {M{\'e}nard}, Brice and {Metcalfe}, Nigel and {Mu{\~n}oz-Guti{\'e}rrez}, Andrea and {Najita}, Joan and {Napier}, Kevin and {Narayan}, Gautham and {Newman}, Jeffrey A. and {Nie}, Jundan and {Nord}, Brian and {Norman}, Dara J. and {Olsen}, Knut A.~G. and {Paat}, Anthony and {Palanque-Delabrouille}, Nathalie and {Peng}, Xiyan and {Poppett}, Claire L. and {Poremba}, Megan R. and {Prakash}, Abhishek and {Rabinowitz}, David and {Raichoor}, Anand and {Rezaie}, Mehdi and {Robertson}, A.~N. and {Roe}, Natalie A. and {Ross}, Ashley J. and {Ross}, Nicholas P. and {Rudnick}, Gregory and {Safonova}, Sasha and {Saha}, Abhijit and {S{\'a}nchez}, F. Javier and {Savary}, Elodie and {Schweiker}, Heidi and {Scott}, Adam and {Seo}, Hee-Jong and {Shan}, Huanyuan and {Silva}, David R. and {Slepian}, Zachary and {Soto}, Christian and {Sprayberry}, David and {Staten}, Ryan and {Stillman}, Coley M. and {Stupak}, Robert J. and {Summers}, David L. and {Sien Tie}, Suk and {Tirado}, H. and {Vargas-Maga{\~n}a}, Mariana and {Vivas}, A. Katherina and {Wechsler}, Risa H. and {Williams}, Doug and {Yang}, Jinyi and {Yang}, Qian and {Yapici}, Tolga and {Zaritsky}, Dennis and {Zenteno}, A. and {Zhang}, Kai and {Zhang}, Tianmeng and {Zhou}, Rongpu and {Zhou}, Zhimin},
        title = "{Overview of the DESI Legacy Imaging Surveys}",
      journal = {\aj},
     keywords = {catalogs, surveys, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2019,
        month = may,
       volume = {157},
       number = {5},
          eid = {168},
        pages = {168},
          doi = {10.3847/1538-3881/ab089d},
        archivePrefix = {arXiv},
          eprint = {1804.08657},
    primaryClass = {astro-ph.IM},
          adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....157..168D},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }


        """,
    "swift_sne_ia": """
        @ARTICLE{2014Ap&SS.354...89B, 
            author = {{Brown}, Peter J. and {Breeveld}, Alice A. and {Holland}, Stephen and {Kuin}, Paul and {Pritchard}, Tyler},
            title = "{SOUSA: the Swift Optical/Ultraviolet Supernova Archive}", 
            journal = {pss}, 
            keywords = {Supernovae, Ultraviolet, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Cosmology and Nongalactic Astrophysics}, 
            year = 2014, 
            month = nov, 
            volume = {354}, 
            number = {1}, 
            pages = {89-96}, 
            doi = {10.1007/s10509-014-2059-8}, 
            archivePrefix = {arXiv}, 
            eprint = {1407.3808}, 
            primaryClass = {astro-ph.HE}, 
            adsurl = {https://ui.adsabs.harvard.edu/abs/2014Ap&SS.354...89B}, 
            adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
          }


        @article{BROWN2015111,
            abstract = {The Swift Gamma Ray Burst Explorer has proven to be an incredible platform for studying the multiwavelength properties of supernova explosions. In its first ten years, Swift has observed over three hundred supernovae. The ultraviolet observations reveal a complex diversity of behavior across supernova types and classes. Even amongst the standard candle type Ia supernovae, ultraviolet observations reveal distinct groups. When the UVOT data is combined with higher redshift optical data, the relative populations of these groups appear to change with redshift. Among core-collapse supernovae, Swift discovered the shock breakout of two supernovae and the Swift data show a diversity in the cooling phase of the shock breakout of supernovae discovered from the ground and promptly followed up with Swift. Swift observations have resulted in an incredible dataset of UV and X-ray data for comparison with high-redshift supernova observations and theoretical models. Swift's supernova program has the potential to dramatically improve our understanding of stellar life and death as well as the history of our universe.},
            author = {Peter J. Brown and Peter W.A. Roming and Peter A. Milne},
            doi = {https://doi.org/10.1016/j.jheap.2015.04.007},
            issn = {2214-4048},
            journal = {Journal of High Energy Astrophysics},
            note = {Swift 10 Years of Discovery, a novel approach to Time Domain Astronomy},
            pages = {111-116},
            title = {The first ten years of Swift supernovae},
            url = {https://www.sciencedirect.com/science/article/pii/S2214404815000178},
            volume = {7},
            year = {2015},
            Bdsk-Url-1 = {https://www.sciencedirect.com/science/article/pii/S2214404815000178},
            Bdsk-Url-2 = {https://doi.org/10.1016/j.jheap.2015.04.007}}

        """,
    "tess": """
        @ARTICLE{2020RNAAS...4..201C, 
          author = {{Caldwell}, Douglas A. and {Tenenbaum}, Peter and {Twicken}, Joseph D. and {Jenkins}, Jon M. and {Ting}, Eric and {Smith}, Jeffrey C. and {Hedges}, Christina and {Fausnaugh}, Michael M. and {Rose}, Mark and {Burke}, Christopher}, 
          title = "{TESS Science Processing Operations Center FFI Target List Products}", 
          journal = {Research Notes of the American Astronomical Society}, 
          keywords = {Catalogs, CCD photometry, Stellar photometry, 205, 208, 1620, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics}, 
          year = 2020, 
          month = nov, 
          volume = {4}, 
          number = {11}, 
          eid = {201}, 
          pages = {201}, 
          doi = {10.3847/2515-5172/abc9b3}, 
          archivePrefix = {arXiv}, 
          eprint = {2011.05495}, 
          primaryClass = {astro-ph.EP}, 
          adsurl = {https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..201C}, 
          adsnote = {Provided by the SAO/NASA Astrophysics Data System} 
        }


        """,
    "vipers": """
        @article{scodeggio2018vimos, 
          title={The VIMOS Public Extragalactic Redshift Survey (VIPERS)-Full spectroscopic data and auxiliary information release (PDR-2)}, 
          author={Scodeggio, MARCO and Guzzo, L and Garilli, BIANCA and Granett, BR and Bolzonella, M and De La Torre, S and Abbas, U and Adami, C and Arnouts, S and Bottini, D and others}, 
          journal={Astronomy & Astrophysics}, 
          volume={609}, 
          pages={A84}, 
          year={2018}, 
          publisher={EDP Sciences}, 
          url={https://arxiv.org/abs/1611.07048} 
        }

        @ARTICLE{2014A&A...562A..23G,
       author = {{Garilli}, B. and {Guzzo}, L. and {Scodeggio}, M. and {Bolzonella}, M. and {Abbas}, U. and {Adami}, C. and {Arnouts}, S. and {Bel}, J. and {Bottini}, D. and {Branchini}, E. and {Cappi}, A. and {Coupon}, J. and {Cucciati}, O. and {Davidzon}, I. and {De Lucia}, G. and {de la Torre}, S. and {Franzetti}, P. and {Fritz}, A. and {Fumana}, M. and {Granett}, B.~R. and {Ilbert}, O. and {Iovino}, A. and {Krywult}, J. and {Le Brun}, V. and {Le F{\`e}vre}, O. and {Maccagni}, D. and {Ma{\l}ek}, K. and {Marulli}, F. and {McCracken}, H.~J. and {Paioro}, L. and {Polletta}, M. and {Pollo}, A. and {Schlagenhaufer}, H. and {Tasca}, L.~A.~M. and {Tojeiro}, R. and {Vergani}, D. and {Zamorani}, G. and {Zanichelli}, A. and {Burden}, A. and {Di Porto}, C. and {Marchetti}, A. and {Marinoni}, C. and {Mellier}, Y. and {Moscardini}, L. and {Nichol}, R.~C. and {Peacock}, J.~A. and {Percival}, W.~J. and {Phleps}, S. and {Wolk}, M.},
        title = "{The VIMOS Public Extragalactic Survey (VIPERS). First Data Release of 57 204 spectroscopic measurements}",
      journal = {\aap},
     keywords = {galaxies: distances and redshifts, galaxies: statistics, galaxies: fundamental parameters, cosmology: observations, catalogs, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2014,
        month = feb,
       volume = {562},
          eid = {A23},
        pages = {A23},
          doi = {10.1051/0004-6361/201322790},
        archivePrefix = {arXiv},
               eprint = {1310.1008},
         primaryClass = {astro-ph.CO},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2014A&A...562A..23G},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        @ARTICLE{2014A&A...566A.108G,
               author = {{Guzzo}, L. and {Scodeggio}, M. and {Garilli}, B. and {Granett}, B.~R. and {Fritz}, A. and {Abbas}, U. and {Adami}, C. and {Arnouts}, S. and {Bel}, J. and {Bolzonella}, M. and {Bottini}, D. and {Branchini}, E. and {Cappi}, A. and {Coupon}, J. and {Cucciati}, O. and {Davidzon}, I. and {De Lucia}, G. and {de la Torre}, S. and {Franzetti}, P. and {Fumana}, M. and {Hudelot}, P. and {Ilbert}, O. and {Iovino}, A. and {Krywult}, J. and {Le Brun}, V. and {Le F{\`e}vre}, O. and {Maccagni}, D. and {Ma{\l}ek}, K. and {Marulli}, F. and {McCracken}, H.~J. and {Paioro}, L. and {Peacock}, J.~A. and {Polletta}, M. and {Pollo}, A. and {Schlagenhaufer}, H. and {Tasca}, L.~A.~M. and {Tojeiro}, R. and {Vergani}, D. and {Zamorani}, G. and {Zanichelli}, A. and {Burden}, A. and {Di Porto}, C. and {Marchetti}, A. and {Marinoni}, C. and {Mellier}, Y. and {Moscardini}, L. and {Nichol}, R.~C. and {Percival}, W.~J. and {Phleps}, S. and {Wolk}, M.},
                title = "{The VIMOS Public Extragalactic Redshift Survey (VIPERS). An unprecedented view of galaxies and large-scale structure at 0.5 < z < 1.2}",
              journal = {\aap},
             keywords = {cosmology: observations, large-scale structure of Universe, galaxies: distances and redshifts, galaxies: statistics, Astrophysics - Cosmology and Nongalactic Astrophysics},
                 year = 2014,
                month = jun,
               volume = {566},
                  eid = {A108},
                pages = {A108},
                  doi = {10.1051/0004-6361/201321489},
        archivePrefix = {arXiv},
               eprint = {1303.2623},
         primaryClass = {astro-ph.CO},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2014A&A...566A.108G},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }




        """,
    "yse": """
        @dataset{aleo_2022_7317476, 
        author = {Aleo, Patrick D. and Malanchev, Konstantin and Sharief, Sammy N. and Jones, David O. and Narayan, Gautham and Ryan, Foley J. and Villar, V. Ashley and Angus, Charlotte R. and Baldassare, Vivienne F. and Bustamante-Rosell, Maria. J. and Chatterjee, Deep and Cold, Cecilie and Coulter, David A. and Davis, Kyle W. and Dhawan, Suhail and Drout, Maria R. and Engel, Andrew and French, K. Decker and Gagliano, Alexander and Gall, Christa and Hjorth, Jens and Huber, Mark E. and Jacobson-Galan, Wynn V. and Kilpatrick, Charles D. and Langeroodi, Danial and Macias, Phillip and Mandel, Kaisey S. and Margutti, Raffaella and Matasic, Filip and McGill, Peter and Pierel, Justin D. R. and Ransome, Conor L. and Rojas-Bravo, Cesar and Siebert, Matthew R. and Smith, Ken W and de Soto, Kaylee M. and Stroh, Michael C. and Tinyanont, Samaporn and Taggart, Kirsty and Ward, Sam M. and Wojtak, Radosław and Auchettl, Katie and Blanchard, Peter K. and de Boer, Thomas J. L. and Boyd, Benjamin M. and Carroll, Christopher M. and Chambers, Kenneth C. and DeMarchi, Lindsay and Dimitriadis, Georgios and Dodd, Sierra A. and Earl, Nicholas and Farias, Diego and Gao, Hua and Gomez, Sebastian and Grayling, Matthew and Grillo, Claudia and Hayes, Erin E. and Hung, Tiara and Izzo, Luca and Khetan, Nandita and Kolborg, Anne Noer and Law-Smith, Jamie A. P. and LeBaron, Natalie and Lin, Chien C. and Luo, Yufeng and Magnier, Eugene A. and Matthews, David and Mockler, Brenna and O'Grady, Anna J. G. and Pan, Yen-Chen and Politsch, Collin A. and Raimundo, Sandra I. and Rest, Armin and Ridden-Harper, Ryan and Sarangi, Arkaprabha and Schrøder, Sophie L. and Smartt, Stephen J. and Terreran, Giacomo and Thorp, Stephen and Vazquez, Jason and Wainscoat, Richard and Wang, Qinan and Wasserman, Amanda R. and Yadavalli, S. Karthik and Yarza, Ricardo and Zenati, Yossef}, 
        title = {{The Young Supernova Experiment Data Release 1 (YSE DR1) Light Curves}}, 
        month = nov, 
        year = 2022, 
        publisher = {Zenodo}, 
        version = {1.0.0}, 
        doi = {10.5281/zenodo.7317476}, 
        url = {https://doi.org/10.5281/zenodo.7317476} 
      }
        """,
}