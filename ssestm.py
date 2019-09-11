import os
import tqdm
import fire
import logging
import string
import re
import csv
import pickle
import math

from scipy.stats import rankdata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import numpy as np


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)


class SSESTM:
    def __init__(self, path="./data", alpha_plus=0.2, alpha_minus=0.2, kappa=3,
                 reg=0.001, alpha_rate=0.0001, max_iters=1000, error=0.000001, skip_params_gen=True):
        self.path = path
        self.skip_params_gen = skip_params_gen

        # Hyper parameters for init
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.kappa = kappa

        # Parameters for data loading
        self.codes = []
        self.returns = []
        self.articles = []
        self.articles_words = []
        self.word_set = []

        # Screened words
        self.S = []
        self.S_count = []

        # P/N Topic parameter for words
        self.O = []

        self.d = []

        # Tuning parameter
        self.reg = reg
        self.alpha_rate = alpha_rate
        self.max_iters = max_iters
        self.error = error

    def _load_data(self):
        logging.info("Loading data...")
        codes = []
        returns = []
        articles = []
        with open(os.path.join(self.path, "data.csv"), "r") as fin:
            rdr = csv.reader(fin)
            for idx, line in tqdm(enumerate(rdr)):
                if idx == 0:
                    continue
                if idx == 200:
                    break
                codes.append(line[0])
                returns.append(float(line[1]))
                articles.append(line[2])
        self.codes = codes
        self.returns = returns
        self.articles = articles

    def _preprocess(self):
        logging.info("Preprocessing...")
        articles_words = []
        for article in tqdm(self.articles):
            articles_words.append(self.preprocess_article(article))
        self.articles_words = articles_words
        word_set = []
        for article_words in articles_words:
            for word in article_words:
                if word not in word_set:
                    word_set.append(word)
        self.word_set = word_set

    def _screen(self):
        logging.info("Screening positive/negative words...")
        for word in tqdm(self.word_set):
            total_cnt = 0
            positive_cnt = 0
            for jdx, article_words in enumerate(self.articles_words):
                if word in article_words:
                    total_cnt += 1
                    if self.returns[jdx] > 0:
                        positive_cnt += 1
            if total_cnt > self.kappa:
                fj = positive_cnt / total_cnt
                if fj >= 1/2 + self.alpha_plus:
                    # Positive sentiment terms
                    self.S.append(word)
                    self.S_count.append(total_cnt)
                elif fj <= 1/2 - self.alpha_minus:
                    # Negative sentiment terms
                    self.S.append(word)
                    self.S_count.append(total_cnt)

    def _calc_topic(self):
        logging.info("Calculating topic matrix...")
        p = []
        n = len(self.articles_words)
        for rank in rankdata(self.returns):
            p.append(rank / n)
        W = np.array([p, [1 - pi for pi in p]], dtype=float)
        WWT = W.dot(W.T)

        d = []
        for idx, article_words in enumerate(self.articles_words):
            di = []
            for word in self.S:
                di.append(article_words.count(word) / self.S_count[idx])
            d.append(np.array(di, dtype=float))
        d = np.array(d, dtype=float)
        O = np.matmul(d.T.dot(W.T), np.linalg.inv(WWT))
        O[O < 0] = 0
        d = np.sum(d, axis=0)
        self.d = np.array(d, dtype=float)

        O_renormalized = []
        for row in O:
            l1_norm = np.sum(row)
            O_renormalized.append(row / l1_norm)
        self.O = np.array(O_renormalized)

    def _save_params(self):
        logging.info("Saving params...")
        with open('./data/O.pickle', 'wb') as f:
            pickle.dump(self.O, f, pickle.HIGHEST_PROTOCOL)

        with open('./data/S.pickle', 'wb') as f:
            pickle.dump(self.S, f, pickle.HIGHEST_PROTOCOL)

        with open('./data/d.pickle', 'wb') as f:
            pickle.dump(self.d, f, pickle.HIGHEST_PROTOCOL)

    def _load_params(self):
        with open('./data/O.pickle', 'rb') as f:
            self.O = pickle.load(f)

        with open('./data/S.pickle', 'rb') as f:
            self.S = pickle.load(f)

        with open('./data/d.pickle', 'rb') as f:
            self.d = pickle.load(f)

    def _predict(self, article):
        p = np.random.rand(1, 1)[0][0]

        preprocessed_article = self.preprocess_article(article)

        s_hat = 0
        s_words = []
        d_article = []
        for word in self.S:
            if word in preprocessed_article:
                d_article.append(preprocessed_article.count(word))
                s_words.append(word)
                s_hat += 1
        d_article = np.array(d_article, dtype=float) * 1 / s_hat

        prev_cost = -1
        for itr in range(self.max_iters):
            cost = self._cost(s_words, p, d_article)
            p = p - self.alpha_rate * self._gradient(s_words, p, d_article)
            logging.info(f"cost: {cost}, p: {p}")

            if prev_cost == -1:
                prev_cost = cost
                continue

            if prev_cost - cost < self.error:
                logging.info(f"Converges to solution")
                break
            prev_cost = cost

    def _cost(self, s_words, p, d_article):
        total_sum = 0
        for idx, word in enumerate(s_words):
            j = self.S.index(word)
            dj = self.d[j]
            total_sum += dj * math.log(p * self.O[j][0] + (1-p) * self.O[j][1]) - d_article[idx]
        return -total_sum - self.reg * math.log(p * (1-p))

    def _gradient(self, s_words, p, d_article):
        total_sum = 0
        for word in s_words:
            j = self.S.index(word)
            dj = self.d[j]
            print(word, " ", dj)
            total_sum += dj * self.O[j][0] / (self.O[j][0] * p + self.O[j][1] * (1 - p))
        return -total_sum - self.reg / (1-p)**2

    def preprocess_article(self, article):
        # Remove numbers
        article = re.sub(r'\d+', '', article.lower())

        # Remove punctuations, symbols
        article = article.translate(str.maketrans('', '', string.punctuation))
        article = " ".join(article.split())

        # Remove stopwords, symbols
        stop_words = set(stopwords.words("english"))
        additional_symbols = ["—", "”", "’", "“", "″"]
        word_tokens = word_tokenize(article)
        word_tokens = [word for word in word_tokens if word not in stop_words and word not in additional_symbols]

        # Lemmatize word tokens
        lemmatizer = WordNetLemmatizer()
        word_tokens = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]

        # Stem word tokens
        stemmer = PorterStemmer()
        word_tokens = [stemmer.stem(word) for word in word_tokens]

        return [word for word in word_tokens if len(word) > 1]

    def run(self):
        logging.info("Running SSESTM")
        if not self.skip_params_gen:
            self._load_data()
            self._preprocess()
            self._screen()
            self._calc_topic()
            self._save_params()
        self._load_params()
        test_text = """
        PART I

Item 1. Business:

        International Business Machines Corporation (IBM or the company) was incorporated in the State of New York on June 16, 1911, as the Computing-Tabulating-Recording Co. (C-T-R), a consolidation of the Computing Scale Co. of America, the Tabulating Machine Co. and The International Time Recording Co. of New York. Since that time, IBM has focused on the intersection of business insight and technological innovation, and its operations and aims have been international in nature. This was signaled over 90 years ago, in 1924, when C-T-R changed its name to International Business Machines Corporation. And it continues today—the company creates value for clients by providing integrated solutions and products that leverage: data, information technology, deep expertise in industries and business processes, with trust and security and a broad ecosystem of partners and alliances. IBM solutions typically create value by enabling new capabilities for clients that transform their businesses and help them engage with their customers and employees in new ways. These solutions draw from an industry-leading portfolio of consulting and IT implementation services, cloud, digital and cognitive offerings, and enterprise systems and software which are all bolstered by one of the world's leading research organizations.

IBM Strategy

        IBM's strategy is wholly focused on the needs of its clients. IBM is a technology company, but first and foremost it is an enterprise company. IBM serves enterprises of all sizes, and IBM's longest-standing clients are leaders in their industries—the world's leading financial services institutions, airlines, manufacturers, consumer goods and retail companies. IBM's mission is to help its clients transform their companies and lead in their industries.

        One of the biggest priorities for IBM clients is to derive competitive advantage through insights and the latest digital technologies. Better insight about the wants and needs of their customers will help them distinguish themselves in the marketplace. Data-driven insight will also influence how they design and produce their own products, as well as help them identify opportunities in new markets.

        However, most companies are harnessing only a small percent of the valuable data they collect. As IBM clients embark on the next chapter of their digital journey, the proper collection, use, safeguarding and management of data is of paramount importance. Choosing the right digital technologies to analyze the data is also necessary.

        IBM helps clients harness the power of their data through technologies like AI, analytics and blockchain; on a hybrid cloud that connects data across traditional and new environments; with services that put a client's data and insight to use in and for their business. Underpinning all of this, IBM safeguards client data with world-class technologies and approaches to security.

        By reinventing themselves digitally around insight, clients become what IBM calls Cognitive Enterprises.

What IBM Brings to Clients

        Businesses are choosing IBM because they want to partner with a company that can uniquely integrate three core capabilities:

1.
They want the most innovative technology, like AI, blockchain, cybersecurity and quantum delivered in a hybrid cloud environment.

2.
They want industry expertise—from a partner that deeply understands their industry and can apply innovation to their business processes to drive transformation and competitive advantage.
1

3.
And, finally, they want a total commitment to trust and security. Clients want to partner with a company that will protect their valuable data and insights, and one that develops and deploys new innovations with a commitment to do so responsibly.
        IBM is unique in that it can integrate all three core capabilities for clients.

Innovative Technology

        IBM has a long history of bringing innovative technology to the world. For 26 years, IBM has led the world in U.S. patents; six IBMers are Nobel Laureates; and IBM engineers have developed innumerable first-of-its-kind products and services. Current examples of IBM's innovative technology include:

•
Analytics and AI: IBM's long-standing leadership in managing and extracting insights from data starts with a portfolio of analytics and database offerings. A few years ago, IBM brought AI into the mainstream with the Watson platform, which to date has been the foundation of many Enterprise AI implementations in production. Recently, IBM augmented its Watson platform with a set of AI tools that enable clients to trace the origins of the data their AI models use, explain what is behind their recommendations and ensure that bias has not crept into results. These innovations are making AI more consumable by everyday users—not just data scientists.

•
Security: Businesses built around data require an unparalleled level of data security. IBM is the leader in information security for enterprises—with leadership in both security software and services. Security is embedded inside all of IBM's products and services. For example, IBM Z offers pervasive security by building data encryption directly in its computing processor. In addition, IBM's Services businesses are world class in embedding security into the solutions they build and run for IBM clients.

•
Blockchain is an exciting technology that is just beginning to transform business processes. IBM's platform has been rated number one by leading analyst firms such as Juniper Research and Everest Group. Blockchain technology enables multiple parties to conduct business with each other on a single, unified distributed system, eliminating the costly and time-consuming hand-offs of fragmented systems. IBM is deploying blockchain technology with clients to transform how global trade is transacted, how food safety is tracked and how supply chains are managed.

•
Cloud: Enterprise clients are in the very early stages of the move to cloud. IBM estimates that only 20 percent of workloads have moved to the cloud—with work ahead for the remaining 80 percent. The first part was to move business workloads that exist as a layer over core processes. The hard part is ahead: moving the mission-critical systems that run banking, retail, telecom and other industries. Some of these workloads will remain in traditional IT systems, some will move to a private cloud inside the safety of a client's firewall, others will move to public clouds, and some will surge between all of these. Wherever a workload may reside, it will need to share its data across environments. All of this requires an approach that is open, highly interoperable between environments, and even interoperable between different public clouds. This is what IBM has long called hybrid cloud—and this describes the solution for the 80 percent of the workloads that is to come. With in-depth experience across all three environments, IBM brings the strongest hybrid cloud solution to the market for enterprises—which will be strengthened through the acquisition of Red Hat, Inc. (Red Hat).
Industry Expertise

        Changing a business requires in-depth understanding of how a business works and how technology can make it work differently.

2

        IBM brings both industry expertise and innovative technology to clients through the IBM services and products businesses. This combination makes IBM unique and essential.

        A few examples of this capability are highlighted below:

•
Global Business Services: the IBM GBS business is one of the world's largest professional services businesses. Its mission is to help clients along the journey to becoming a Cognitive Enterprise.

•
Global Technology Services: the IBM GTS business runs some of the world's largest data centers—and thereby some of the world's most mission-critical workflows and franchises. GTS helps clients along their journey to the hybrid cloud—leveraging the best of their existing systems in the context of the regulatory, security and workflow of their industry.

•
Industry and Domain-Specific Solutions: augmenting IBM's services businesses are software and solutions designed for specific industries and domains. For example:
Health: IBM has become a leader in applying advanced digital technologies to healthcare, including the application of AI and data analytics to the diagnosis and treatment of patients, bringing smart decisions to Health Care payers, and helping Life Sciences companies develop innovative products and services.

Financial Services: IBM is a leading provider in the Financial Services industry; with IBM's Promontory Financial Group, a leading advisor in Financial Regulation and Compliance, IBM offers an advanced set of solutions for managing Risk and Compliance, a critical workflow in the Financial Services industry.

Trust and Security

        Data and AI—together, they are both the opportunity and the issue of current times. They can make the world a better, healthier and more productive place; but only if businesses and consumers trust the companies putting data and AI to work.

        IBM is a 107-year old business—and the reason it has been successful for so long is because it has earned the trust of its clients. IBM has not only followed guidelines around the responsible handling of data and the stewardship of new technology, but created them, published them, and invited others to adopt similar commitments. IBM's principles make clear that:

•
The purpose of new technologies is to augment—not replace—human expertise;

•
Data and insights derived from AI belong to their owners and creators (not their IT partners); and,

•
New technologies must be transparent and explainable.
        There are many companies in the IT industry who bring technology products to the marketplace. Many bring technology services to the marketplace. A few companies do both, but no one can do it as well as IBM when it comes to meeting the needs of clients. By bringing together technology and workflow, combining it with industry expertise, innovation and deployment, IBM helps clients and industries truly transform themselves.

        This is what truly sets IBM apart.

Business Model

        The company's business model is built to support two principal goals: helping enterprise clients to move from one era to the next by bringing together innovative technology and industry expertise, and providing long-term value to shareholders. The business model has been developed over time through

3

strategic investments in capabilities and technologies that have long-term growth and profitability prospects based on the value they deliver to clients.

        The company's global capabilities include services, software, systems, fundamental research and related financing. The broad mix of businesses and capabilities are combined to provide integrated solutions and platforms to the company's clients.

        The business model is dynamic, adapting to the continuously changing industry and economic environment, including the company's transformation into cloud and as-a-Service delivery models. The company continues to strengthen its position through strategic organic investments and acquisitions in higher-value areas, broadening its industry expertise and integrating AI into more of what the company offers. In addition, the company is transforming into a more agile enterprise to drive innovation and speed, as well as helping to drive productivity, which supports investments for participation in markets with significant long-term opportunity. The company also regularly evaluates its portfolio and proactively maximizes shareholder value of non-strategic assets by bringing products to end of life, engaging in IP partnerships or executing divestitures.

        This business model, supported by the company's financial model, has enabled the company to deliver strong earnings, cash flows and returns to shareholders over the long term.

Business Segments and Capabilities

        The company's major operations consist of five business segments: Cognitive Solutions, Global Business Services, Technology Services & Cloud Platforms, Systems and Global Financing.

        Cognitive Solutions comprises a broad portfolio of primarily software capabilities that help IBM's clients to identify actionable new insights and inform decision-making for competitive advantage. Leveraging IBM's research, technology and industry expertise, this business delivers a full spectrum of capabilities, from descriptive, predictive and prescriptive analytics to artificial intelligence. Cognitive Solutions includes Watson, the first enterprise AI platform that specializes in driving value and knowledge from the 80 percent of the world's data that sits behind company firewalls. It enables businesses to reimagine their workflows across a variety of industries and professions and gives organizations complete control of their insights, data, training and IP.

        Additionally, Cognitive Solutions includes the new Watson OpenScale technology—a first of a kind, open technology platform that addresses key challenges of AI adoption. It enables companies to manage AI transparently throughout the full AI lifecycle, irrespective of where their AI applications were built or in which environment they currently run.

        IBM's solutions are provided through the most contemporary delivery methods including through cloud environments and "as-a-Service" models. Cognitive Solutions consists of Solutions Software and Transaction Processing Software.

Cognitive Solutions Capabilities

        Solutions Software: provides the basis for many of the company's strategic areas. IBM has established the world's deepest portfolio of enterprise AI, including analytics and data management platforms, cloud data services, talent management solutions, and solutions tailored by industry. Watson Platform, Watson Health and Watson Internet of Things (IoT) are certain capabilities included in Solutions Software. IBM's world-class security platform weaves in AI to deliver integrated security intelligence across clients' entire operations, including their cloud, applications, networks and data, helping them to prevent, detect and remediate potential threats.

        Transaction Processing Software: includes software that primarily runs mission-critical systems in industries such as banking, airlines and retail.

4

        Global Business Services (GBS) provides clients with consulting, application management and business process services. These professional services deliver value and innovation to clients through solutions which leverage industry, technology and business strategy and process expertise. GBS is the digital reinvention partner for IBM clients, combining industry knowledge, functional expertise, and applications with the power of business design and cognitive and cloud technologies. The full portfolio of GBS services is backed by its globally integrated delivery network and integration with technologies, solutions and services from IBM units including IBM Watson, IBM Cloud, IBM Research, and Global Technology Services.

        In 2018, focused on digital reinvention, GBS assisted clients on their journeys to becoming Cognitive Enterprises, helping them engage their customers with new digital value propositions, transform workflows using AI, and build hybrid, open cloud infrastructures. This was delivered by the operating model rolled out in 2017—Digital Strategy and iX, Cognitive Process Transformation and Cloud Application Innovation, cross industry and globally.

GBS Capabilities

        Consulting: provides business consulting services focused on bringing to market solutions that help clients shape their digital blueprints and customer experiences, define their cognitive operating models, unlock the potential in all data to improve decision-making, set their next-generation talent strategies and create new technology architectures in a cloud-centric world.

        Application Management: delivers system integration, application management, maintenance and support services for packaged software, as well as custom and legacy applications. Value is delivered through advanced capabilities in areas such as security and privacy, application testing and modernization, cloud application migration and automation.

        Global Process Services (GPS): delivers finance, procurement, talent and engagement, and industry-specific business process outsourcing services. These services deliver improved business results to clients through a consult-to-operate model which includes the strategic change and/or operation of the client's processes, applications and infrastructure. GBS is redefining process services for both growth and efficiency through the application of the power of cognitive technologies like Watson, as well as the IoT, blockchain and deep analytics.

        Technology Services & Cloud Platforms (TS&CP) provides comprehensive IT infrastructure and platform services that create business value for clients. Clients gain access to leading-edge, high-quality services, flexibility and economic value. This is enabled through leverage of insights drawn from IBM's decades of experience across thousands of engagements, the skills of practitioners, advanced technologies, applied innovation from IBM Research and global scale.

TS&CP Capabilities

        Infrastructure Services: delivers a portfolio of project services, managed and outsourcing services and cloud-delivered services focused on clients' enterprise IT infrastructure environments to enable digital transformation and deliver improved quality, flexibility and economic value. The portfolio includes a comprehensive set of hybrid cloud services and solutions to assist enterprise clients in building and running contemporary IT environments. These offerings integrate long-standing expertise in service management and technology with the ability to utilize the power of new technologies, drawn from across IBM's businesses and ecosystem partners. The portfolio is built using the IBM Services Platform with Watson, designed to augment human intelligence with cognitive technologies, and addresses hybrid cloud, digital workplace, business resiliency, network, managed applications, cloud and security. The company's capabilities, including IBM Cloud, cognitive computing and hybrid cloud implementation, provide high-performance, end-to-end innovation and an improved ability for clients to achieve business objectives.

5

        Technical Support Services: delivers comprehensive support services to maintain and improve the availability of clients' IT infrastructures. These offerings include maintenance for IBM products and other technology platforms, as well as open source and vendor software and solution support, drawing on innovative technologies and leveraging the IBM Services Platform with Watson capabilities.

        Integration Software: delivers industry-leading hybrid cloud solutions that empower clients to achieve rapid innovation, hybrid integration, and process transformation with choice and consistency across public, dedicated and local cloud environments, leveraging the IBM Platform-as-a-Service solution. Integration Software offerings and capabilities help clients address the digital imperatives to create, connect and optimize their applications, data and infrastructure on their journey to become cognitive businesses.

        Systems provides clients with innovative infrastructure platforms to help meet the new requirements of hybrid cloud and enterprise AI workloads. More than one-third of Systems Hardware's server and storage sales transactions are through the company's business partners, with the balance direct to end-user clients. IBM Systems also designs advanced semiconductor and systems technology in collaboration with IBM Research, primarily for use in the company's systems.

Systems Capabilities

        Systems Hardware: includes IBM's servers: IBM Z, Power Systems and Storage Systems.

        Servers: a range of high-performance systems designed to address computing capacity, security and performance needs of businesses, hyperscale cloud service providers and scientific computing organizations. The portfolio includes IBM Z, a trusted enterprise platform for integrating data, transactions and insight, and Power Systems, a system designed from the ground up for big data and enterprise AI, optimized for hybrid cloud and Linux, and delivering open innovation with OpenPOWER.

        Storage Systems: data storage products and solutions that allow clients to retain and manage rapidly growing, complex volumes of digital information and to fuel data-centric cognitive applications. These solutions address critical client requirements for information retention and archiving, security, compliance and storage optimization including data deduplication, availability and virtualization. The portfolio consists of a broad range of flash storage, disk and tape storage solutions.

        Operating Systems Software: IBM Z operating system environments include z/OS, a security-rich, high-performance enterprise operating system, as well as Linux. Power Systems offers a choice of AIX, IBM i or Linux operating systems. These operating systems leverage POWER architecture to deliver secure, reliable and high performing enterprise-class workloads across a breadth of server offerings.

        Global Financing encompasses two primary businesses: financing, primarily conducted through IBM Credit LLC (IBM Credit), and remanufacturing and remarketing. IBM Credit is a wholly owned subsidiary of IBM that accesses the capital markets directly. IBM Credit, through its financing solutions, facilitates IBM clients' acquisition of information technology systems, software and services in the areas where the company has expertise. The financing arrangements are predominantly for products or services that are critical to the end users' business operations. The company conducts a comprehensive credit evaluation of its clients prior to extending financing. As a captive financier, Global Financing has the benefit of both deep knowledge of its client base and a clear insight into the products and services financed. These factors allow the business to effectively manage two of the major risks associated with financing, credit and residual value, while generating strong returns on equity. Global Financing also maintains a long-term partnership with the company's clients through various stages of the IT asset life cycle—from initial purchase and technology upgrades to asset disposition decisions.

6

Global Financing Capabilities

        Client Financing: lease, installment payment plan and loan financing to end users and internal clients for terms up to seven years. Assets financed are primarily new and used IT hardware, software and services where the company has expertise. Internal financing is predominantly in support of Technology Services & Cloud Platforms' long-term client service contracts. All internal financing arrangements are at arm's-length rates and are based upon market conditions.

        Commercial Financing: short-term working capital financing to suppliers, distributors and resellers of IBM and Original Equipment Manufacturer (OEM) products and services. The OEM portion will begin winding down starting in the second quarter of 2019 and continuing throughout the calendar year. Commercial Financing also includes internal activity where Global Financing factors a selected portion of the company's accounts receivable primarily for cash management purposes, at arm's-length rates.

        Remanufacturing and Remarketing: assets include used equipment returned from lease transactions, or used and surplus equipment acquired internally or externally. These assets may be refurbished or upgraded, and sold or leased to new or existing clients both externally or internally. Externally remarketed equipment revenue represents sales or leases to clients and resellers. Internally remarketed equipment revenue primarily represents used equipment that is sold internally to Systems and Technology Services & Cloud Platforms. Systems may also sell the equipment that it purchases from Global Financing to external clients.

IBM Worldwide Organizations

        The following worldwide organizations play key roles in IBM's delivery of value to its clients:

•
Global Markets

•
Research, Development and Intellectual Property
Global Markets

        IBM has a global presence, operating in more than 175 countries with a broad-based geographic distribution of revenue. The company's Global Markets organization manages IBM's global footprint, working closely with dedicated country-based operating units to serve clients locally. These country teams have client relationship managers who lead integrated teams of consultants, solution specialists and delivery professionals to enable clients' growth and innovation.

        By complementing local expertise with global experience and digital capabilities, IBM builds deep and broad-based client relationships. This local management focus fosters speed in supporting clients, addressing new markets and making investments in emerging opportunities. The Global Markets organization serves clients with expertise in their industry as well as through the products and services that IBM and partners supply. IBM continues to expand its reach to new and existing clients through digital marketplaces, digital sales and local Business Partner resources.

7


Research, Development and Intellectual Property

        IBM's research and development (R&D) operations differentiate the company from its competitors. IBM annually invests approximately 7 percent of total revenue for R&D, focusing on high-growth, high-value opportunities. IBM Research works with clients and the company's business units through global labs on near-term and mid-term innovations. It delivers many new technologies to IBM's portfolio every year and helps clients address their most difficult challenges. IBM Research scientists are conducting pioneering work in artificial intelligence, quantum computing, blockchain, security, cloud, nanotechnology, silicon and post-silicon computing architectures and more—applying these technologies across industries including financial services, healthcare, blockchain and IoT.

        In 2018, for the 26th consecutive year, IBM was awarded more U.S. patents than any other company. IBM's 9,100 patents awarded in 2018 represent a diverse range of inventions in strategic growth areas for the company, including more than 3,000 patents related to work in artificial intelligence, cloud, cybersecurity and quantum computing.

        The company actively continues to seek IP protection for its innovations, while increasing emphasis on other initiatives designed to leverage its IP leadership. Some of IBM's technological breakthroughs are used exclusively in IBM products, while others are licensed and may be used in IBM products and/or the products of the licensee. As part of its business model, the company licenses certain of its intellectual property assets, which constitute high-value technology, but may be applicable in more mature markets. The licensee drives the future development of the IP and ultimately expands the customer base. This generates IP income for the company both upon licensing, and with any ongoing royalty arrangements between it and the licensee. While the company's various proprietary IP rights are important to its success, IBM believes its business as a whole is not materially dependent on any particular patent or license, or any particular group of patents or licenses. IBM owns or is licensed under a number of patents, which vary in duration, relating to its products.

Competition

        The company is a globally-integrated enterprise, operating in more than 175 countries. The company participates in a highly competitive environment, where its competitors vary by industry segment, and range from large multinational enterprises to smaller, more narrowly focused entities. Overall, across its business segments, the company recognizes hundreds of competitors worldwide.

        Across its business, the company's principal methods of competition are: technology innovation; performance; price; quality; brand; its broad range of capabilities, products and services; client relationships; the ability to deliver business value to clients; and service and support. In order to maintain leadership, a corporation must continue to invest, innovate and integrate. Over the last several years, the company has been making investments and shifting resources, embedding AI and cloud into its offerings while building new solutions and modernizing its existing platforms. These investments not only drive current performance, but will extend the company's innovation leadership into the future. The company's key differentiators are built around three pillars–innovative technology, industry expertise and trust and security, uniquely delivered through an integrated model. As the company executes its strategy, it has entered into new markets, such as hybrid cloud, digital, analytics, AI, blockchain and quantum, and deployed new delivery models, including as-a-service solutions, each of which expose the company to new competitors. Overall, the company is the leader or among the leaders in each of its business segments.

8

        A summary of the competitive environment for each business segment is included below:

Cognitive Solutions:

        The Cognitive Solutions segment leads the burgeoning market for artificial intelligence infused software solutions. Increasingly, technology companies are looking to implement software solutions that will take advantage of the massive amounts of data businesses hold in order to improve business outcomes for their clients. The Watson platform is integrated throughout the Cognitive Solutions portfolio. Watson is IBM's suite of enterprise-ready AI services, applications and tooling—built specifically for business. Delivered through the cloud, the platform analyzes data, understands complex questions posed in natural language, and proposes evidence-based answers. Watson continuously learns in three ways: by being taught by its users, by learning from prior interactions, and by being presented with new information. Watson specializes in small data and driving value and knowledge from the 80 percent of the world's data that sits behind company firewalls. It enables businesses to reimagine their workflows across a variety of industries and professions and gives organizations control of their insights, data, training and IP.

        The segment's key competitive factors include a wide range of powerful Watson AI services—machine learning to deep learning. IBM is unique in that it allows clients to retain ownership of their data, protect insights and ensures AI transparency and trust. It trains with small specialized data sets, and is focused on embedding AI into business workflows. IBM's AI systems are trained and designed for specific industries and professions including health, financial services, education, retail, agriculture, supply chain, human resources, marketing, advertising, and more.

        Cognitive Solutions includes solutions software, delivered both on-premise and "as-a-Service", and transaction processing software. The solutions software portfolio, which spans data management, analytics, security and social capabilities, provides comprehensive business and industry-specific offerings to IT decision makers. IT buyers include chief information officers as well as line of business buyers, such as chief marketing and procurement officers, chief information security officers and chief financial officers. The transaction processing software portfolio, mostly delivered on-premise, runs mission-critical systems in industries such as banking, airlines and retail.

        The depth and breadth of the software offerings, coupled with the company's global markets and technical support infrastructure, differentiate its capabilities from its competitors. The company's research and development capabilities and intellectual property patent portfolio also contribute to its differentiation. The company's principal competitors in this segment include Alphabet Inc. (Google), Amazon.com, Inc. (Amazon), Cisco Systems, Inc. (Cisco), Microsoft Corporation (Microsoft), Oracle Corporation (Oracle) and SAP. The company also competes with smaller, niche competitors in specific geographic or product markets.

Global Business Services and Technology Services & Cloud Platforms:

        The company's services segments, Global Business Services and Technology Services & Cloud Platforms, operate in a highly competitive and continually evolving global market. Competitive factors in these business segments include: quality of services, innovative offerings, financial value, technical skills and capabilities, industry knowledge and experience and speed of execution. The company's competitive advantages in these businesses comes from its ability to design, implement, manage and support integrated solutions that address complex client needs across hybrid cloud environments, leveraging automation, AI, extensive expertise in technology and innovation, services assets, and a strong set of relationships with clients and strategic business partners worldwide.

9

Global Business Services:

        GBS competes in consulting, systems integration, application management and business process outsourcing services. The company competes with broad based competitors including: Accenture, Capgemini, DXC Technology (DXC), Fujitsu, Google and Microsoft; India-based service providers; the consulting practices of public accounting firms; and many companies that primarily focus on local markets or niche service areas.

Technology Services & Cloud Platforms:

        Technology Services & Cloud Platforms competes in project services, managed and outsourcing services, cloud-delivered services, and a wide range of technical and IT support services. The company competes with IT service providers including: Atos, DXC, Fujitsu, HCL, Tata Consulting Services, Wipro and many companies that primarily focus on local markets or niche service areas. The company also competes with cloud platform vendors such as Amazon, Google, Microsoft and Oracle.

        This segment also includes the company's Integration Software offerings. Integration Software helps clients address the digital imperatives to create, connect and optimize their applications, data and infrastructure on their journey to become cognitive businesses. The company competes with Amazon, BMC, Microsoft, Oracle and VMWare, as well as companies that primarily focus on niche solutions and offerings.

Systems:

        The enterprise server and storage market is characterized by competition in technology and service innovation focused on value, function, reliability, price and cost performance. The company's principal competitors include Dell Technologies, Hewlett-Packard Enterprise (HPE), Intel and lower cost original device manufacturer systems that are often re-branded. Also, alternative as-a-service providers are leveraging innovation in technology and service delivery both to compete with traditional providers and to offer new routes to market for server and storage systems. These alternative providers include Amazon, Google, Microsoft, and IBM's own cloud-based services.

        The company gains advantage and differentiation through investments in higher value capabilities—from semiconductor through software stack innovation—that increase efficiency, lower cost and improve performance. The company's research and development capabilities and intellectual property patent portfolio contribute significantly to this segment's leadership across areas as diverse as high performance computing, virtualization technologies, software optimization, power management, security, multi-operating system capabilities and open technologies like interconnect standards to be leveraged by broad ecosystems.

Global Financing:

        Global Financing provides client financing, commercial financing and participates in the remarketing of used equipment. Global Financing's access to capital and its ability to manage credit and residual value risk generates a competitive advantage for the company. The key competitive factors include interest rates charged, IT product experience, client service, contract flexibility, ease of doing business, global capabilities and residual values. In client and commercial financing, Global Financing competes with three types of companies in providing financial services to IT customers: other captive financing entities of IT companies such as Cisco and HPE, non-captive financing entities and banks or financial institutions. In remarketing, the company competes with local and regional brokers plus original manufacturers in the fragmented worldwide used IT equipment market.

10

Forward-looking and Cautionary Statements

        Certain statements contained in this Form 10-K may constitute "forward-looking statements" within the meaning of the Private Securities Litigation Reform Act of 1995 ("Reform Act"). Forward-looking statements are based on the company's current assumptions regarding future business and financial performance. These statements by their nature address matters that are uncertain to different degrees. The company may also make forward-looking statements in other reports filed with the Securities and Exchange Commission (SEC), in materials delivered to stockholders and in press releases. In addition, the company's representatives may from time to time make oral forward-looking statements. Forward-looking statements provide current expectations of future events based on certain assumptions and include any statement that does not directly relate to any historical or current fact. Words such as "anticipates," "believes," "expects," "estimates," "intends," "plans," "projects," and similar expressions, may identify such forward-looking statements. Any forward-looking statement in this Form 10-K speaks only as of the date on which it is made. The company assumes no obligation to update or revise any forward-looking statements. In accordance with the Reform Act, set forth under Item 1A. "Risk Factors" on pages 12 to 18 are cautionary statements that accompany those forward-looking statements. Readers should carefully review such cautionary statements as they identify certain important factors that could cause actual results to differ materially from those in the forward-looking statements and from historical trends. Those cautionary statements are not exclusive and are in addition to other factors discussed elsewhere in this Form 10-K, in the company's filings with the SEC or in materials incorporated therein by reference.

        The following information is included in IBM's 2018 Annual Report to Stockholders and is incorporated herein by reference:

        Segment information and revenue by classes of similar products or services—pages 141 to 146.

        Financial information regarding environmental activities—page 111.

        The number of persons employed by the registrant—page 67.

        The management discussion overview—pages 19 to 21.

        Website information and company reporting—page 150.

Executive Officers of the Registrant (at February 26, 2019):

 
 	Age	 	Officer since	 
Virginia M. Rometty, Chairman of the Board, President and Chief Executive Officer*

 	 	61	 	 	2005	 
Michelle H. Browdy, Senior Vice President, Legal and Regulatory Affairs, and General Counsel

 	 	54	 	 	2015	 
Erich Clementi, Senior Vice President

 	 	60	 	 	2010	 
Robert F. Del Bene, Vice President and Controller

 	 	59	 	 	2017	 
Diane J. Gherson, Senior Vice President and Chief Human Resources Officer

 	 	61	 	 	2013	 
James J. Kavanaugh, Senior Vice President and Chief Financial Officer, Finance and Operations

 	 	52	 	 	2008	 
John E. Kelly III, Executive Vice President

 	 	65	 	 	2000	 
Kenneth M. Keverian, Senior Vice President, Corporate Strategy

 	 	62	 	 	2014	 
Martin J. Schroeter, Senior Vice President, Global Markets, Global Financing, Marketing, and Communications

 	 	54	 	 	2014	 
*
Member of the Board of Directors.
        All executive officers are elected by the Board of Directors annually as provided in the Company's By-laws. Each executive officer named above, with the exception of Kenneth M. Keverian, has been an executive of IBM or its subsidiaries during the past five years.

11

        Mr. Keverian was a Senior Partner at the Boston Consulting Group, a global management consulting firm, until joining IBM in 2014. He was with Boston Consulting Group for 26 years and he focused on serving technology companies in the computing and communications sectors.
        """
        self._predict(test_text)

if __name__ == '__main__':
    fire.Fire(SSESTM)
