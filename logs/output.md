```
python code\business_entity_resolution\src\eda.py
```

```
==================== 1. ROW COUNTS ====================
train S1:  2,206,821    test S1:  1,732,544
train S2:  5,034,616    test S2:  4,887,273
train S3:  5,285,603    test S3:  5,082,316
ground truth rows: 2,206,821

==================== 2. COUNTRY COUNTS ====================
train S1: {'US': 1323633, 'India': 883188}
train S2: {'US': 3016817, 'India': 2017799}
train S3: {'US': 3170056, 'India': 2115547}
test S1: {'India': 809986, 'US': 663106, 'France': 259452}
test S2: {'India': 2312565, 'US': 1871330, 'France': 703378}
test S3: {'India': 2405000, 'US': 1945701, 'France': 731615}

==================== 3. EMPTY RATE PER COLUMN (%) ====================
train S1: {'entity_id': np.float64(0.0), 'business_name': np.float64(0.0), 'business_address': np.float64(0.0), 'country': np.float64(0.0)}
train S2: {'entity_id': np.float64(0.0), 'business_name': np.float64(0.0), 'business_address': np.float64(3.36), 'country': np.float64(0.0)}
train S3: {'entity_id': np.float64(0.0), 'business_name': np.float64(0.0), 'business_address': np.float64(3.33), 'country': np.float64(0.0)}
test S1: {'entity_id': np.float64(0.0), 'business_name': np.float64(0.0), 'business_address': np.float64(0.0), 'country': np.float64(0.0)}
test S2: {'entity_id': np.float64(0.0), 'business_name': np.float64(0.0), 'business_address': np.float64(2.65), 'country': np.float64(0.0)}
test S3: {'entity_id': np.float64(0.0), 'business_name': np.float64(0.0), 'business_address': np.float64(2.68), 'country': np.float64(0.0)}

==================== 4. GROUND TRUTH STATS ====================
S1 entities in GT: 2,206,821  (train S1 rows: 2,206,821)
singleton rate: 5.58%
matches per S1 (total):
n
0     123247
1     119157
2     375212
3     530841
4     484115
5     321957
6     164868
7      63968
8      18680
9       4205
10       534
11        37
matches per S1 from S2: {0: 287745, 1: 789108, 2: 652779, 3: 333957, 4: 119078, 5: 24154}
matches per S1 from S3: {0: 266276, 1: 716417, 2: 668375, 3: 372443, 4: 145116, 5: 35378, 6: 2816}
total matched S2 ids: 3,693,619 / 5,034,616 S2 rows  (73.4% of S2 are matched)
total matched S3 ids: 3,944,746 / 5,285,603 S3 rows  (74.6% of S3 are matched)
duplicate S2/S3 ids across S1 groups: 0
GT ids missing from train S2/S3 files: 0

by country (singleton %, mean matches):
            rows  singleton_pct  mean_matches
country
India     883188           5.59      3.464543
US       1323633           5.58      3.459057

==================== 5. DUPLICATE NAMES ====================
S1: unique names 1,539,229/2,206,821  exact (name,addr) dups: 0
S2: unique names 4,402,009/5,034,616  exact (name,addr) dups: 25,891
S3: unique names 4,651,609/5,285,603  exact (name,addr) dups: 18,881

==================== 6. NAME / ADDRESS LENGTH (chars) ====================
S1 name len: {'mean': 24.0, '50%': 24.0, 'max': 105.0}  addr len: {'mean': 52.1, '50%': 41.0, 'max': 256.0}
S2 name len: {'mean': 25.1, '50%': 25.0, 'max': 104.0}  addr len: {'mean': 46.2, '50%': 37.0, 'max': 249.0}
S3 name len: {'mean': 25.2, '50%': 25.0, 'max': 123.0}  addr len: {'mean': 46.7, '50%': 42.0, 'max': 240.0}

==================== 7. NON-ASCII AND POSTAL CODE PRESENCE (train, % of rows) ====================
S1 India: non-ascii name 0.00%   postal code in address 0.0%
S1 US: non-ascii name 0.00%   postal code in address 10.8%
S2 India: non-ascii name 27.87%   postal code in address 0.0%
S2 US: non-ascii name 6.70%   postal code in address 10.1%
S3 India: non-ascii name 18.49%   postal code in address 0.0%
S3 US: non-ascii name 6.80%   postal code in address 10.2%
TEST S1 France: rows 259,452  non-ascii name 15.72%  5-digit code in addr 0.4%
TEST S2 France: rows 703,378  non-ascii name 24.54%  5-digit code in addr 0.5%
TEST S3 France: rows 731,615  non-ascii name 23.94%  5-digit code in addr 0.5%

==================== 8. SAMPLE MATCHED GROUPS (train) ====================

----- India -----

[S1-672772660] Fortune Finance Private Limited | F.No.202, Aranaya Balbharda East B.C.Road, Patna-1, Patna, Bihar
    -> S2-38561496: फॉर्च्यून फाइनेंस प्राइवेट लिमिटेड | F.NO.202, ARANAYA BALBHARDA EAST B.C.ROAD, PATNA-1, PATNA, Bihar
    -> S2-34503666: fortunefinance.com | F.NO.202 , ARANAYA BALBHARDA EAST B.C.ROAD, PATNA-1, Bihar
    -> S3-321479436: Fortune Finance Private Limited | F.no.202, Aranaya Balbharda East B.c.road, Patna-1, Patna, BR
    -> S3-244510480: फॉर्च्यून फाइनेंस प्राइवेट लिमिटेड | F.no.202, Patna-1, Patna, BR

[S1-91111365] Perfect Consultants Limited | H.No.1/133, Palaprassery, Chengamanad P.O, Ernakulam, Paravur, Ernakulam, Kerala
    -> S2-664580510: Limited Perfect Cotnsuhftanst | PARAVUR, ERNAKULAM, Kerala, H.NO.1/133
    -> S3-956219693: Dr Perfect  Limited Center | H.no.1/133, Keralam
    -> S3-665007251: പെർഫെക്റ്റ് Consultants ലിമിറ്റഡ് | H.no.1/133, Ernakulam, Keralam
    -> S3-488174891: പെർഫെക്റ്റ് കൺസൾട്ടന്റ്സ് ലിമിറ്റഡ് | H.no.1/133, Paravur, Keralam

[S1-638204887] Anand Ventures Pvt Ltd | House No 121, Second Floor, Rohini Block-E, Pkt-19, Sec-3, New Delhi, North West, Delhi
    -> S2-594201036: आनंद वेंचर्स प्रा. लि. | HOUSE NO 121, SECOND FLOOR, ROHINI BLOCK-E, PKT-19, SEC-3, NORTH WEST, Delhi
    -> S2-506793163: Anand Ventures Ltd Pvt | #867 HOUSE NO 121, SECOND FLOOR, ROHINI BLOCK-E, PKT-19, SEC-3, NEW DELHI, Delhi
    -> S2-94806574: ANAND VÉNTURES PVT LTD | 867 HOUSE NO 121, SECOND FLOOR, ROHINI BLOCK-E, PKT-19, SEC-3, NEW DELHI, Delhi
    -> S2-16598681: आनंद वेंचर्स प्रा. लि. | Delhi, #867 HOUSE NO 121, NEW DELHI, NORTH WEST
    -> S3-167923976: आनंद Ventures प्रा. लि. | House No 121, North West, DL

[S1-26292216] Yadav Trading Private Limited | 515, Kmj Identity Terrace, Floor, 20Th Main Road, Bangalore South, Bangalore, Karnataka
    -> S2-997560673: PRIVATEYADAVTRADINGCOM | 515, KMJ IDENTITY TERRACE, FLOOR, 20TH MAIN ROAD, BANGALORE SOUTH, KORAMANGALA, ಕರ್ನಾಟಕ
    -> S3-381438957: Yadav Trading Private Ltd | #B3/560 515, Kmj Identity Terrace, Floor, 20Th Main Road, Bangalore, KA

[S1-942463527] Al Media Private Limited | 78A Raja Basant Roy Road, 2Nd Floor, Kolkata, Howrah, West Bengal
    -> S2-940874238: আল মিডিয়া প্রাইভেট লিমিটেড | West Bengal, HOWRAH, #78A RAJA BASANT ROY ROAD, KOLKATA
    -> S2-6956757: Smt Al Media  Private Ltd | 78A RAJA BASANT ROY ROAD, 2ND FLOOR, KOLKATA, পশ্চিমবঙ্গ
    -> S2-897408378: Al Media Private | H.NO 78A RAJA BASANT ROY ROAD, 2ND FLOOR, KOLKATA, পশ্চিমবঙ্গ
    -> S2-321668623: AL MEDIA PRIVATE | H.NO 78A RAJA BASANT ROY ROAD, 2ND FLOOR, KOLKATA, West Bengal
    -> S3-725251917: Al Media Private Ltd | 78A Raja Basant Roy Road, Kolkata, Howrah, পশ্চিমবঙ্গ

[S1-731840942] Dhanalakshmi Forex Associates | C/O Angu Srinu Kiranreddy, Tarlapad, Khanapur, Adilabad, Telangana
    -> S2-364389978: dhanalakshmi forex associates | C/O ANGU SRINU KIRANREDDY, TARLAPAD, KHANAPUR, Telangana
    -> S2-965144164: Dhanalakshmi Forex Associates | C/O ANGU SRINU KIRANREDDY, TARLAPAD, KHANAPUR, తెలంగాణ
    -> S2-916842861: Sri Dhanalakshmi  Forex | C/O ANGU SRINU KIRANREDDY, TARLAPAD, KHANAPUR, Telangana
    -> S3-748897478: Dhanalakshmi Forex  Associates | C/o Angu Srinu Kiranreddy, Khanapur, Adilabad, తెలంగాణ
    -> S3-3317997: dhanalakshmi forex associates | C/o Angu Srinu Kiranreddy, Adilabad, TG

[S1-725664088] Team Nirman Private Limited | Shop At C 1 A 51 A, G F, Janak Puri, New Delhi, West Delhi, Delhi
    -> S2-442904262: Private Team Nirman Ltd | NEW DELHI, WEST DELHI, SHOP AT C 1 A 51 A, दिल्ली
    -> S3-688280841: Private Team Nirman Ltd | Shop At C B3/1 A 51 A, West Delhi, New Delhi, DL
    -> S3-824148013: Team Nirman Private Ltd | Shop At C B3/1 A 51 A, G F, Janak Puri, New Delhi, DL

[S1-385355733] Dynamic Solutions Private Limited | 1378, Saphalya, Near Bhavani Mandap Shanti Nagar, Nachane, Ratnagiri, Maharashtra
    -> S2-303182115: डायनामिक सॉल्यूशंस प्राइवेट लिमिटेड | RATNAGIRI, Maharashtra, RATNAGIRI, SAPHALYA, NEAR BHAVANI MANDAP SHANTI NAGAR, NACHANE
    -> S3-866975212: Dynamic Solutions Private [Limited] | 1378, Goa-panaji Region, Ratnagiri, MH
    -> S3-64115388: डायनामिक सॉल्यूशंस प्राइवेट लिमिटेड | 1378, Goa-panaji Region, Ratnagiri, MH

----- US -----

[S1-67993537] Prairie Islands | 930 Park Central Drive, Unit C, Indianapolis, IN
    -> S2-850642616: Prairie Islands | 00930 PARK CENTRAL DRIVE, INDIANAPOLIS, IN
    -> S2-562664581: Prairie Islands  Inc. | 00930 PARK CENTRAL DRIVE, INDIANAPOLIS, IN
    -> S3-891114156: Prairie Islands | 930. Park Central Drive, # C, Indianapolis, Indiana
    -> S3-108446678: Prairie Íslands Íslands | 930. Park Central Dr, Unit C, Indianapolis, Indiana

[S1-527441188] Coastal Avalanche Co | 109 Park Avenue, Unit D, Sugar Grove, IL
    -> S2-347262887: Coastal Co Avalanche | 109 PARK AVE, SUGAR GROVE, IL
    -> S3-90533948: Coastal Avalanche  Co | 109 Park Avenue, # D, Sugar Grove, Illinois
    -> S3-227078595: Coastal  Aalainsche Co | Illinois, # D, Sugar Grove, 109 Park Avenue

[S1-764602504] Dorelle's Ace Cargo LP | 4009 47th Street, MO, Kansas City
    -> S2-909808094: Dorelle's Ace Cargo LP | ##3852 47TH SAINT, KANSAS CITY, MO
    -> S3-421343887: Dorelle's Ace Cargo (LLC) | 4009 47th Street, Kansas City, Missouri
    -> S3-493295011: DAC | 4009 47th Street, Kansas City, MO

[S1-291788566] Riordan Ridge Liquor LLC | 2254 Big Ridge Road, Dickenson County, VA
    -> S3-676115047: Riordan-Ridge Liquor | Big Ridge Rd, Dickenson County, Virginia

[S1-644446863] Pittsboro Scholarship Trust | 108 Haven Creek Road, Unit 205, Pittsboro, NC
    -> S2-309615070: Pittsboro Scho1arship Trust Corp | 108-B HAVEN CREEK RD, PITTSBORO, NC
    -> S2-699749072: Pittsboro Scholarship Trust-Corporation |
    -> S2-600159298: Pittsboro Scohlahohip Trust | 108-B HAVEN CREEK ROAD, PITTSBORO, NC
    -> S3-740712704: Pittsboro Scholarship | Pittsboro, North Carolina, # 205, 108 Haven Creek Road

[S1-325014597] Walker Hall of Malden P.C. | 9 Wyeth Street, Malden, MA
    -> S2-945418047: WALKER HALL OF MAMDLEN P.C. | 9 WYETH SAINT, MADEN, MA
    -> S2-24093714: Walker Hall Of Malden P.C. | 9 WYETH STREET, MA, MADEN
    -> S3-287031643: Walker Hall of Malden | 9 Wyeth Street, Malden, Massachusetts
    -> S3-7429478: LLC Walker Hall of Malden | Malden, 9 Wyeth St, Massachusetts

[S1-391049372] Standard Automation Worldwide PLLC | 10140 White Dog Way, Flagstaff, AZ
    -> S2-936793274: Standard Automation PLLC Worldwide | 10140-C WHITE DOG WAY, FLAGSTAFF, AZ
    -> S3-579927378: Standard Automation Worldwide Pllc | 10140 White Dog Way, Doney Park, Arizona

[S1-273636412] The Dent Auto Body! | NC, 115 Atchison Street, Garner
    -> S2-143128595: The Dent Auto Body! Center | 115-119 ATCHISON STREET, GARNER, NC
    -> S3-569323420: dentauto.com | 115-117 Atchison St, Garner CITY, North Carolina

==================== 9. SAMPLE SINGLETONS (train) ====================
[S1-726638459] US | Orthopedic Health Inc. | 494 Champion Avenue, Unit C, Columbus, OH
[S1-942851466] US | Kestola Joint LLC | 1400 Water Street, Jamesville, NC
[S1-759473212] India | Af Electronics Private Limited | 402, Forth Floor, Kakrola Housing Complex, New Delhi, South West Delhi, Delhi
[S1-414334260] India | Patna Sadar Safety Pvt Ltd | Flat No- 101, Alba Orchid, Alba Colony, Patna Sadar, Patna, Bihar
[S1-15385337] India | Jai Energy Private Limited | Plot No 6 Gut No 151 Near Maharana Pratap School Vijayanth Nagar Deolai Road Satara Paris, Ar, Aurangabad, Maharashtra
[S1-663602102] India | Chennai Suburban Private Limited | No.72, Sivaprakasam Nagar Surapet, Ambattur, Chennai, Tamil Nadu
[S1-818090404] US | Complete Scientific LLC | 962 Healing Springs Drive, Denton, NC
[S1-587898141] India | Best Healthcare Limited | D-354 Sector 47 Noida, Gautam Budh Nagar, Gautam Buddha Nagar, Uttar Pradesh
[S1-133898544] US | Averill & Carr | 7900 Woodville Road, Mount Airy, MD
[S1-874555723] US | Precision Service PC | Lowell, MA, 16 Manning Drive

==================== 10. UNMATCHED S2/S3 ROWS (distractors), sample ====================

S2: 1,340,997 unmatched rows
  S2-520944191 | US | corp solik arrived | 31 PLEASANT ST, BOSTON, MA
  S2-286651992 | US | Christiansen Preferred Furniture, Partners | 7928 ASHLAND AVENUE, CHCIAGO, IL
  S2-137902261 | US | Chambers, Begic and Na Alussa West Inc. | 380 ESCONDIDA DR, TOMBSTONNE, AZ
  S2-726381982 | US | NEXDISOFT CORP  SERVICE #98428 | 576 POPLAR AVE, COLLIERVILE CITY, TN
  S2-994182191 | US | Eye Prime Associates Inc | TN, MT JULIET, 32 Schooner Ln
  S2-865456699 | India | Upadhyay Producer Pvt Ltd | NO. 45 547, MAHANA SINGH ROAD BAGH JALLIAN WALA, AMRITSAR, Punjab

S3: 1,340,857 unmatched rows
  S3-16761311 | India | Hv Land Exports Pvt | No. ##14 W 18 Teaknit Industrial Complexbehind Instrumentation Ltd Kanjikode West, Kottayam, Palghat, KL
  S3-952040986 | US | Walton Lease Partners Co | 29 Guadalupe Dr, Fort Worth, Texas
  S3-603418692 | US | Legacy Star  Crypto Riverside, Inc. | www.legacysta.com | 16923-C Kyler Creek Trail, Humble, Texas
  S3-559389422 | India | हरि Products प्राइवेट लिमिटेड | Thane, Navi Mumbai Region, MH, C 23
  S3-592075027 | US | Shrader And Turner Southside Southside Corp | 887 Yacht Club Dr, Ogden Valley, Utah
  S3-799008382 | US | Canton Holdings | 7198 Robertson Rd, Nashville, Indiana

==================== 11. S1 ROWS SHARING A NAME WITH ANOTHER S1 ROW (hard negatives) ====================
S1 rows sharing a normalized name: 870,894 / 2,206,821
   entity_id     business_name                                       business_address country
S1-509657161       007 Capital                3902 Fenwick Street, Henrico County, VA      US
S1-664015615       007 Capital Unit 304, Hendersonville, TN, 1531 Hunt Club Boulevard      US
S1-640556352 007 Costamare LLC                                    425 812, Higdon, AL      US
S1-112067562 007 Costamare LLC                           66 Mill Lane, Huntington, NY      US
S1-631925158 007 Costamare LLC            6626 Belleview Street, Fl 1, Scottsdale, AZ      US
S1-815155654       007 Devices           114 1/2 Ozark Street, Unit APT 4, Girard, KS      US
S1-855497724       007 Devices                  2336 Carruthers Drive, Fort Worth, TX      US
S1-871639319       007 Devices                    3196 Choccolocco Road, Anniston, AL      US
S1-301459852        007 Golden                       127 Hubert Boulevard, Hubert, NC      US
S1-139449660        007 Golden                      1120 Sycamore, Unit 221, Mesa, AZ      US
S1-325438183       007 Lending                 3700 Duchess Drive, City Of Racine, WI      US
S1-511892623       007 Lending                                255 690, Smithfield, UT      US

==================== 12. GROUPS WITH BOTH S2 AND S3 MATCHES ====================
S1 groups with >=1 S2 and >=1 S3 match: 1,776,047 (80.5%)
```

```
python code\business_entity_resolution\src\eda2.py
```

```
==================== A. FRANCE: raw samples (test) ====================
S1 | Fédération de Vos | Hauts-de-France, 46 Rue Maximilien de Robespierre, Lille
S1 | Boxing Theatre SARL | 2 Stade du Souvenir, Calais, Hauts-de-France
S1 | Vercors (France) Maternelle SARL | 170 Rue Desaugiers, Lille, Hauts-de-France
S1 | Paradis & Frères EURL | 19 b Rue Francon, La Teste-de-Buch, Nouvelle-Aquitaine
S1 | Information & Frères SARL | 39 Quai Lucien Lheureux, Calais, Hauts-de-France
S1 | Racing Solidarite Etablissement SASU | 24 Avenue des Pinsons, La Teste-de-Buch, Nouvelle-Aquitaine
S1 | Autonomie (France) Amicale SA | 41 Rue de Neuville, Tourcoing, Hauts-de-France
S1 | Recherche College SAS | 36 Rue Commandant Hautreux, Bordeaux, Nouvelle-Aquitaine
S1 | Sos Lycee (France) SAS | 3 Rue Jeanne d'Arc, Nantes, Pays de la Loire
S1 | Tennis Primaire SAS | 29 Chemin de Launay Violette, Pays de la Loire, Nantes
S1 | Centre Hospitalier Souris | 696 Avenue Émile Loubet, Dunkerque, Hauts-de-France
S1 | Archers Centre SAS | Saint-Nazaire, 79 Rue de la Croix Fraîche, Pays de la Loire
S1 | Rencontres Maison SAS | 3 TER Rue de Lompret, Lille, Hauts-de-France
S1 | Hospitaliers Lune Club SAS | 44 Avenue du Maréchal de Lattre de Tassigny (rd), Pessac, Nouvelle-Aquitaine
S1 | Bordeaux Groupe EI | 34 Rue Frédéric Bastiat, Bordeaux, Nouvelle-Aquitaine

-- S2 France sample --
S2 | LTO Parents Développement SASU | 8 RUE DUCAU, BORDEAUX, Nouvelle-Aquitaine
S2 | Union de Amap SCI | Gironde, LÈGE-CAP-FERRET, 24 ALLEE DES DUNES
S2 | SA Quimper Atelier | Pornic, Loire-Atlantique, 0012 LA BOUTEILLERE
S2 | sarl douillet street club groupe | 24 R LOUIS JOUVET, Nantes
S2 | DROITS PHÂRMACIE FRANCE SARL | 6 RUE DE WATTIGNIES, TOURCOING, Nord
S2 | VD EURL Services |
S2 | Maison de  Santé Defense | 142 R. DE LA LIBERTE, BORDEAUX, Gironde
S2 | Dunkerque Club SARL | Hauts-de-France, DUNKERQUE, N° 14 RUE DES SCIERIES
S2 | Amicale des Tarot SCI | 290 AVENUE D'ARES, MERIGNAC, Nouvelle-Aquitaine
S2 | Roubaix Centre SA | 50 RUE DU CHEAVLIER ROLAND, ROUBAIX
S2 | Association du Fil SAS | 60 LA DEVAIRIE, PORNIC, Pays de la Loire
S2 | Amicale De Garde | N° 27 M RUE MARIE DUFAURE, LA TESTE-DE-BUCH, Nouvelle-Aquitaine
S2 | ucentrafricaine.com | N°25 RUE GODEFROY CAVAIGNAC, Lille, Nord
S2 | Amicale dë Herbes | 3 BIS RUE DU PIN, NANTES, Loire-Atlantique
S2 | Maison Departemental Groupe | 40 RUE DES SARRAZINS, LILLE, Hauts-de-France

-- S3 France sample --
S3 | Centre Médical Sainte Justice | (15) Rue De Caudéran, Bordeaux
S3 | Ets Residents-S.A.S.U. | 53 R De La Tanuerie, Calais, Pas-de-Calais
S3 | SAS Serpent Club (Frânce) | #15 R. Du Moulin Fagot, Tourcoing, Nord
S3 | Cannes &-Fils SARL | 2ter R. André Chenier, Lille
S3 | St Abstract SAS & Associés | 28 Rue Du Capitaine Guynemer, Tourcoing, Hauts-de-France
S3 | UP Centre SARL | 53 Bis R. De Lens, Lille
S3 | Fédération De Sèrre | 51 Rue Camille Godard, Bordeaux, Nouvelle-Aquitaine
S3 | Europraid  (France) | 5 Rue De L’oustalet, La Teste-de-buch
S3 | Lège-Cap-Ferret Hopital | 16 Avenue De L'herbe, Lège-cap-ferret, Nouvelle-Aquitaine
S3 | Centre Petits France | 149 Rue Mattéotti, Lille, Nord
S3 | Cantou Insertion EI France | N° 117 Imp. Isaert, Dunkerque, Hauts-de-France
S3 | Stomp Parents France Sarl | 25 R. De La Misérecorde, Bordeaux, Nouvelle-Aquitaine
S3 | Centre Médical Sainte Pierre S.A.R.L. | 44000 Nantes, Nantes, 026 Rue De Chype
S3 | Danse Paul | 35B Rue Malus, Ecole De Musique Municipale, Lille, Nord
S3 | SABLES PHOTO MAISON | 72 Rue Marcelin Berthlot, Bordeaux, Gironde

==================== B. SCRIPTS IN NAMES: share of rows containing each script ====================
train S1 US (n=200,000): {'latin': '100.00%'}
train S2 US (n=200,000): {'latin': '100.00%', 'latin-ext(accents)': '6.69%'}
train S3 US (n=200,000): {'latin': '99.98%', 'latin-ext(accents)': '6.85%'}
train S1 India (n=200,000): {'latin': '100.00%'}
train S2 India (n=200,000): {'latin': '77.40%', 'devanagari': '13.30%', 'latin-ext(accents)': '4.50%', 'telugu': '1.93%', 'kannada': '1.88%', 'tamil': '1.70%', 'gujarati': '1.56%', 'bengali': '1.53%'}
train S3 India (n=200,000): {'latin': '88.53%', 'devanagari': '7.43%', 'latin-ext(accents)': '5.31%', 'telugu': '1.10%', 'kannada': '1.03%', 'tamil': '0.93%', 'gujarati': '0.88%', 'bengali': '0.87%'}
test S1 France (n=200,000): {'latin': '100.00%', 'latin-ext(accents)': '15.71%'}
test S2 France (n=200,000): {'latin': '100.00%', 'latin-ext(accents)': '24.51%'}
test S3 France (n=200,000): {'latin': '100.00%', 'latin-ext(accents)': '23.95%'}

France accent chars in S1 names (top 12):
[('é', 9678), ('É', 4586), ('è', 3959), ('ç', 40), ('ô', 15), ('ê', 6), ('\x92', 2), ('â', 2), ('ë', 1), ('û', 1)]

France: names that are fully non-latin in S2 (sample):
rows with any non-latin script: 0 / 100,000
Empty DataFrame
Columns: [business_name, business_address]
Index: []

==================== C. BLOCKING PROBE on 100k train S1 (recall of simple keys) ====================
true pairs in probe: 346,376

% of true pairs sharing a key, by country x source:
country           India            US
src                  S2     S3     S2     S3
addr_empty          3.8    3.9    4.9    4.6
addr_tok_shared    96.1   95.7   93.9   94.4
addr_tok_shared2   94.8   86.8   79.1   80.6
num_shared         83.2   81.7   76.3   79.6
name_tok_shared    71.8   80.2   92.0   92.2
name_latin         71.7   80.7   92.5   92.5
any_key           100.0  100.0  100.0  100.0
addr_or_num        96.2   96.1   95.0   95.4
name_first_same    58.0   62.0   83.8   78.8

Overall:
addr_empty            4.4
addr_tok_shared      94.9
addr_tok_shared2     84.2
num_shared           79.8
name_tok_shared      85.7
name_latin           86.0
any_key             100.0
addr_or_num          95.6
name_first_same      72.7

When name is NON-latin (script mismatch), % of pairs with shared addr token / number:
rows: 48,382  addr_tok_shared 96.9%  num_shared 81.9%  addr_or_num 97.4%  addr_empty 2.6%

==================== D. SIMILARITY DISTRIBUTIONS: true pairs vs random same-country pairs ====================
.\code\business_entity_resolution\src\eda2.py:178: UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
  same = T.name_latin = T.mn.map(lambda s: s.isascii())
(latin-name true pairs only, n=17,115)
name token_set  TRUE : [ 22.  57.  91. 100. 100.]
name token_set  RAND : [ 0.  0. 27. 32. 38.]
addr token_set  TRUE : [ 61.  74.  88.  95. 100.]
addr token_set  RAND : [21. 25. 31. 34. 38.]
addr token_set between DIFFERENT S1 entities with the SAME name: [24. 28. 34. 38. 42.] (n=3000)

Done.
```

```
python code\business_entity_resolution\src\eval.py
```

```
0.714 (expected 0.714)
eval self-test OK
```

```
python -c "import sys; sys.path.insert(0,'code/business_entity_resolution/src'); from normalize import *; print(norm_name('Fortune Finance Private Limited')); print(norm_name('fortunefinance.com')); print(norm_name('Cannes &-Fils SARL')); print(norm_addr('F.No.202, Aranaya Balbharda East B.C.Road, Patna-1')); print(norm_addr('44 Avenue du Maréchal de Lattre de Tassigny (rd), Pessac')); print(house_numbers('10140-C WHITE DOG WAY'), house_numbers('00930 PARK CENTRAL DRIVE'))"
```

```
('fortune finance', 'private limited')
('fortunefinance', '')
('cannes and fils', 'sarl')
f number 202 aranaya balbharda east b c road patna 1
44 avenue du marechal de lattre de tassigny road pessac
{'10140'} {'930'}
```

```
python code\business_entity_resolution\src\blocking.py --n 5000
```

```
loading train files ...
loaded in 20s
S1 subsample: 5,000
S1 names: 100%|███████████████████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 322465.13row/s]
S1 addrs: 100%|███████████████████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 178567.65row/s]
S2 names: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:12<00:00, 391808.49row/s]
S2 addrs: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:26<00:00, 190030.09row/s]
S3 names: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:13<00:00, 380809.32row/s]
S3 addrs: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:27<00:00, 192293.16row/s]

=== blocking vs S2 ===

  [S2 | India] queries=2,041 targets=2,017,799
    S2/India/addr-word: fitting TF-IDF on 1,959,953 rows ...
    S2/India/addr-word: fit 10s  vocab=231,893  nnz/row=3.3
    S2/India/addr-word sim+topk: 100%|███████████████████████████████████████████| 2041/2041 [00:00<00:00, 12850.84q/s]
    S2/India/addr-word: peak nnz per chunk = 3,302,602
  [S2 | India] addr-word done 12s
    S2/India/name-word: fitting TF-IDF on 2,017,799 rows ...
    S2/India/name-word: fit 4s  vocab=69,191  nnz/row=1.2
    S2/India/name-word sim+topk: 100%|████████████████████████████████████████████| 2041/2041 [00:00<00:00, 9677.65q/s]
    S2/India/name-word: peak nnz per chunk = 4,254,690
  [S2 | India] name-word done 17s
    S2/India/name-char4: fitting TF-IDF on 2,017,799 rows ...
    S2/India/name-char4: fit 10s  vocab=141,949  nnz/row=4.5
    S2/India/name-char4 sim+topk: 100%|███████████████████████████████████████████| 2041/2041 [00:00<00:00, 5949.53q/s]
    S2/India/name-char4: peak nnz per chunk = 5,490,305
  [S2 | India] name-char4 done 29s

  [S2 | US] queries=2,959 targets=3,016,817
    S2/US/addr-word: fitting TF-IDF on 2,905,696 rows ...
    S2/US/addr-word: fit 9s  vocab=161,423  nnz/row=2.7
    S2/US/addr-word sim+topk: 100%|██████████████████████████████████████████████| 2959/2959 [00:00<00:00, 10328.29q/s]
    S2/US/addr-word: peak nnz per chunk = 3,582,417
  [S2 | US] addr-word done 11s
    S2/US/name-word: fitting TF-IDF on 3,016,817 rows ...
    S2/US/name-word: fit 7s  vocab=103,509  nnz/row=2.1
    S2/US/name-word sim+topk: 100%|███████████████████████████████████████████████| 2959/2959 [00:00<00:00, 3075.72q/s]
    S2/US/name-word: peak nnz per chunk = 11,327,954
  [S2 | US] name-word done 20s
    S2/US/name-char4: fitting TF-IDF on 3,016,817 rows ...
    S2/US/name-char4: fit 21s  vocab=181,503  nnz/row=6.4
    S2/US/name-char4 sim+topk: 100%|██████████████████████████████████████████████| 2959/2959 [00:01<00:00, 2385.53q/s]
    S2/US/name-char4: peak nnz per chunk = 11,027,147
  [S2 | US] name-char4 done 43s

S2: pair recall 0.9058 | S1 fully recalled 0.8648 | avg candidates/S1 34.3 | 74s
   India: pair recall 0.8849 | full 0.8301 | avg cands 34.1
   US: pair recall 0.9205 | full 0.8883 | avg cands 34.5

=== blocking vs S3 ===

  [S3 | India] queries=2,041 targets=2,115,547
    S3/India/addr-word: fitting TF-IDF on 2,050,583 rows ...
    S3/India/addr-word: fit 10s  vocab=207,951  nnz/row=2.9
    S3/India/addr-word sim+topk: 100%|███████████████████████████████████████████| 2041/2041 [00:00<00:00, 11748.38q/s]
    S3/India/addr-word: peak nnz per chunk = 3,497,327
  [S3 | India] addr-word done 11s
    S3/India/name-word: fitting TF-IDF on 2,115,547 rows ...
    S3/India/name-word: fit 4s  vocab=75,789  nnz/row=1.3
    S3/India/name-word sim+topk: 100%|████████████████████████████████████████████| 2041/2041 [00:00<00:00, 9563.70q/s]
    S3/India/name-word: peak nnz per chunk = 3,965,250
  [S3 | India] name-word done 17s
    S3/India/name-char4: fitting TF-IDF on 2,115,547 rows ...
    S3/India/name-char4: fit 12s  vocab=144,727  nnz/row=4.6
    S3/India/name-char4 sim+topk: 100%|███████████████████████████████████████████| 2041/2041 [00:00<00:00, 7326.01q/s]
    S3/India/name-char4: peak nnz per chunk = 4,451,439
  [S3 | India] name-char4 done 30s

  [S3 | US] queries=2,959 targets=3,170,056
    S3/US/addr-word: fitting TF-IDF on 3,059,088 rows ...
    S3/US/addr-word: fit 10s  vocab=163,686  nnz/row=2.7
    S3/US/addr-word sim+topk: 100%|███████████████████████████████████████████████| 2959/2959 [00:00<00:00, 6965.69q/s]
    S3/US/addr-word: peak nnz per chunk = 5,233,925
  [S3 | US] addr-word done 12s
    S3/US/name-word: fitting TF-IDF on 3,170,056 rows ...
    S3/US/name-word: fit 7s  vocab=111,858  nnz/row=2.1
    S3/US/name-word sim+topk: 100%|███████████████████████████████████████████████| 2959/2959 [00:01<00:00, 2377.50q/s]
    S3/US/name-word: peak nnz per chunk = 11,709,153
  [S3 | US] name-word done 21s
    S3/US/name-char4: fitting TF-IDF on 3,170,056 rows ...
    S3/US/name-char4: fit 22s  vocab=181,440  nnz/row=6.6
    S3/US/name-char4 sim+topk: 100%|██████████████████████████████████████████████| 2959/2959 [00:01<00:00, 2222.35q/s]
    S3/US/name-char4: peak nnz per chunk = 11,552,442
  [S3 | US] name-char4 done 46s

S3: pair recall 0.8845 | S1 fully recalled 0.8269 | avg candidates/S1 33.7 | 79s
   India: pair recall 0.8285 | full 0.7424 | avg cands 32.7
   US: pair recall 0.9225 | full 0.8839 | avg cands 34.4
```

```
python code\business_entity_resolution\src\block_diag.py --n 5000
```

```
loading ...
S1 names: 100%|███████████████████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 345017.11row/s]
S1 addrs: 100%|███████████████████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 175253.38row/s]
S2 names: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:12<00:00, 390271.60row/s]
S2 addrs: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:27<00:00, 185459.47row/s]
S3 names: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:18<00:00, 278909.12row/s]
S3 addrs: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:37<00:00, 141279.35row/s]

################ S2 ################
  [S2|India] q=2,041 t=2,017,799
      addr_w.002: nnz/row(t)=3.3 peak_nnz=1,664,058
    addr_w.002 done 15s
      addr_w.02: nnz/row(t)=5.5 peak_nnz=22,156,700
    addr_w.02 done 17s
      name_w.005: nnz/row(t)=1.2 peak_nnz=2,152,772
    name_w.005 done 4s
      name_w.05: nnz/row(t)=1.8 peak_nnz=10,487,443
    name_w.05 done 5s
      name_c4.003: nnz/row(t)=4.5 peak_nnz=2,781,948
    name_c4.003 done 11s
      name_c4.02: nnz/row(t)=8.6 peak_nnz=19,959,294
    name_c4.02 done 15s
      comb_w.02: nnz/row(t)=7.0 peak_nnz=32,174,014
    comb_w.02 done 17s
      comb_w.1: nnz/row(t)=10.0 peak_nnz=181,073,952
    comb_w.1 done 32s
  [S2|US] q=2,959 t=3,016,817
      addr_w.002: nnz/row(t)=2.7 peak_nnz=1,813,477
    addr_w.002 done 11s
      addr_w.02: nnz/row(t)=3.7 peak_nnz=13,584,050
    addr_w.02 done 13s
      name_w.005: nnz/row(t)=2.1 peak_nnz=5,845,440
    name_w.005 done 8s
      name_w.05: nnz/row(t)=2.7 peak_nnz=23,272,005
    name_w.05 done 11s
      name_c4.003: nnz/row(t)=6.4 peak_nnz=5,622,794
    name_c4.003 done 24s
      name_c4.02: nnz/row(t)=13.3 peak_nnz=46,431,690
    name_c4.02 done 36s
      comb_w.02: nnz/row(t)=5.8 peak_nnz=24,059,734
    comb_w.02 done 20s
      comb_w.1: nnz/row(t)=7.6 peak_nnz=141,118,091
    comb_w.1 done 37s

--- S2: single-channel recall@K (all / India / US) ---
channel        K=5             K=10            K=15            K=25            K=40            K=50
addr_w.002     0.716/0.738/0.700 0.761/0.789/0.742 0.780/0.805/0.762 0.803/0.825/0.788 0.821/0.838/0.808 0.829/0.848/0.816
addr_w.02      0.820/0.849/0.799 0.861/0.880/0.847 0.874/0.888/0.864 0.886/0.898/0.877 0.896/0.905/0.889 0.900/0.907/0.895
name_w.005     0.468/0.362/0.543 0.536/0.428/0.612 0.570/0.463/0.646 0.605/0.505/0.676 0.636/0.534/0.708 0.648/0.547/0.719
name_w.05      0.557/0.492/0.602 0.619/0.546/0.670 0.650/0.577/0.702 0.683/0.615/0.731 0.708/0.638/0.758 0.718/0.644/0.770
name_c4.003    0.337/0.279/0.377 0.390/0.342/0.423 0.417/0.379/0.444 0.449/0.415/0.473 0.480/0.454/0.498 0.494/0.470/0.512
name_c4.02     0.518/0.460/0.559 0.581/0.516/0.626 0.610/0.546/0.655 0.647/0.585/0.691 0.676/0.608/0.723 0.687/0.616/0.738
comb_w.02      0.936/0.914/0.951 0.954/0.933/0.969 0.961/0.942/0.974 0.965/0.947/0.978 0.970/0.952/0.983 0.972/0.956/0.984
comb_w.1       0.942/0.922/0.956 0.960/0.940/0.974 0.965/0.946/0.978 0.971/0.954/0.983 0.976/0.960/0.987 0.978/0.964/0.988

--- S2: union configs (recall all / India / US | avg cands/S1) ---
base(addr.002,name.005,c4.003 @15/15/10)     0.9058 / 0.8849 / 0.9205 | 34.3
loose@15                                     0.9583 / 0.9479 / 0.9655 | 37.1
loose@25                                     0.9680 / 0.9577 / 0.9753 | 62.1
loose+comb@25                                0.9769 / 0.9626 / 0.9870 | 73.5
comb.1+addr.02+c4@25                         0.9800 / 0.9666 / 0.9895 | 61.6
loose+comb@40                                0.9847 / 0.9747 / 0.9917 | 127.3

--- S2: miss reasons for widest union (129 missed pairs) ---
India  foreign_script_name                    61  (47.3%)
US     shares_addr_token(rank/crowded)        23  (17.8%)
US     empty_addr                             18  (14.0%)
India  shares_addr_token(rank/crowded)        16  (12.4%)
India  empty_addr                             11  (8.5%)

examples (S1 name | S1 addr || missed name | missed addr):
  [India / foreign_script_name]
     Shiva New Consultancy Private Limited | B/101, Bellona 1St Floor, 90 Feet Road, Pant Nagar, Ghatkopar East, Mumbai, Mumbai City, Maharashtra  ||  शिवा न्यू कंसल्टेंसी प्राइवेट लिमिटेड | B/101, MUMBAI, MUMBAI CITY, महाराष्ट्र
     Shiva New Consultancy Private Limited | B/101, Bellona 1St Floor, 90 Feet Road, Pant Nagar, Ghatkopar East, Mumbai, Mumbai City, Maharashtra  ||  शिवा न्यू कंसल्टेंसी प्राइवेट लिमिटेड | B/101, MUMBAI, MUMBAI CITY, Maharashtra
     Pioneer Technology | A503, Bldg No.4, Stella Sapphire Chembur Crystal Chs Sahakar Nagar Road No 3, Chembur (E), Mumbai City, Mumbai, Maharashtra  ||  पायोनियर टेक्नोलॉजी | A503, BOMBAY, MUMBAI CITY, Maharashtra
  [US / shares_addr_token(rank/crowded)]
     Reliable Audio Installation LLC | 135 Woodland Drive, Wartburg, TN  ||  @RELIABLEAUDIO | 13 WOODLAND DR, WARRTBURG CDP, TN
     Castorena Lithium | 1915 Mission Avenue, Topeka, KS  ||  Vantagecira | KS, TOPEKACDP, 5915 MISSION AVENUE
     Elite Heartland Ave LLC | 321 Green Meadows Drive, Unit D, Gahanna, OH  ||  heavecom | 321 GREEN MEADWS DRIVE, COLUMBUS, OH
  [US / empty_addr]
     Heritage Coalition | 5949 30th Avenue, Portland, OR  ||  Heritage Cd1aftnibon |
     Heartland Airlines Inc | VA, Purcellville Town, 525 Gentlewood Square  ||  Heartland Aierbibes Inc |
     Urology Associates | Winchester, OH, 191 Russellville Road  ||  URO1OGY ASSOCIATES |
  [India / shares_addr_token(rank/crowded)]
     Apex & Sons Private Limited | Bangalore South, 2Nd Floor, 1St Main, Sector-6, Hsr Layout, Bangalore, No. 574/A, Karnataka  ||  Veraavi | NO. 74/A, BANGALORE, BANGALORE SOUTH, Karnataka
     Oswal More (India) Private Limited | Flat No - R-6, Lgf, Ansal Chamber-1, 3 Bhikaji Cama Place, Delhi, New Delhi, Delhi  ||  0swal More (India) Private Ltd | FLAT NO - R-##6, DELHI, NEW DELHI, दिल्ली
     Cheer Vidyalaya Private Limited | H.No.156, Kapil Vihar, Pitampura Opp Metro Pillar 355, Delhi, North West, Delhi  ||  Deltasyn | H.NO.156, NORTH WEST, Delhi
  [India / empty_addr]
     Jai Food LLP | New Delhi, E-286 Grater Kailash-Ii, South Delhi, Delhi  ||  Jai फूड एलएलपी |
     Kozhikode Media Pvt Ltd | 46/130, Alaprath Square, Kundaythode, Kolathara Po, Kozhikode, Kerala  ||  Kozhikode Merda Pvt Ltd |
     Golden Ventures Limited | Plot No. 179, Gidc Estate, Ramangamdi, Vadodara, Gujarat  ||  Golden Limited Center |

################ S3 ################
  [S3|India] q=2,041 t=2,115,547
      addr_w.002: nnz/row(t)=2.9 peak_nnz=1,762,347
    addr_w.002 done 12s
      addr_w.02: nnz/row(t)=5.2 peak_nnz=27,352,720
    addr_w.02 done 15s
      name_w.005: nnz/row(t)=1.3 peak_nnz=2,085,829
    name_w.005 done 6s
      name_w.05: nnz/row(t)=2.1 peak_nnz=12,778,107
    name_w.05 done 7s
      name_c4.003: nnz/row(t)=4.6 peak_nnz=2,243,621
    name_c4.003 done 14s
      name_c4.02: nnz/row(t)=9.4 peak_nnz=21,341,517
    name_c4.02 done 17s
      comb_w.02: nnz/row(t)=6.8 peak_nnz=38,595,946
    comb_w.02 done 19s
      comb_w.1: nnz/row(t)=9.6 peak_nnz=170,454,889
    comb_w.1 done 33s
  [S3|US] q=2,959 t=3,170,056
      addr_w.002: nnz/row(t)=2.7 peak_nnz=2,674,240
    addr_w.002 done 13s
      addr_w.02: nnz/row(t)=3.8 peak_nnz=13,580,238
    addr_w.02 done 15s
      name_w.005: nnz/row(t)=2.1 peak_nnz=6,025,776
    name_w.005 done 10s
      name_w.05: nnz/row(t)=2.8 peak_nnz=24,565,555
    name_w.05 done 13s
      name_c4.003: nnz/row(t)=6.6 peak_nnz=5,928,028
    name_c4.003 done 29s
      name_c4.02: nnz/row(t)=13.7 peak_nnz=49,310,769
    name_c4.02 done 41s
      comb_w.02: nnz/row(t)=6.0 peak_nnz=25,831,698
    comb_w.02 done 23s
      comb_w.1: nnz/row(t)=7.9 peak_nnz=91,636,718
    comb_w.1 done 34s

--- S3: single-channel recall@K (all / India / US) ---
channel        K=5             K=10            K=15            K=25            K=40            K=50
addr_w.002     0.681/0.633/0.714 0.723/0.675/0.755 0.744/0.695/0.777 0.767/0.715/0.802 0.785/0.734/0.820 0.795/0.743/0.829
addr_w.02      0.765/0.733/0.787 0.812/0.772/0.839 0.830/0.790/0.857 0.849/0.806/0.878 0.864/0.821/0.893 0.870/0.829/0.898
name_w.005     0.427/0.305/0.510 0.499/0.367/0.587 0.533/0.401/0.622 0.567/0.436/0.655 0.598/0.469/0.685 0.614/0.489/0.699
name_w.05      0.522/0.457/0.566 0.591/0.517/0.642 0.625/0.549/0.676 0.662/0.592/0.710 0.697/0.634/0.739 0.714/0.655/0.753
name_c4.003    0.312/0.253/0.351 0.357/0.299/0.395 0.384/0.334/0.418 0.417/0.375/0.445 0.446/0.410/0.471 0.461/0.423/0.486
name_c4.02     0.483/0.418/0.527 0.549/0.473/0.600 0.582/0.503/0.635 0.620/0.543/0.673 0.658/0.595/0.701 0.677/0.617/0.717
comb_w.02      0.915/0.866/0.948 0.940/0.896/0.970 0.950/0.910/0.977 0.958/0.924/0.981 0.964/0.937/0.983 0.968/0.944/0.984
comb_w.1       0.919/0.872/0.951 0.946/0.906/0.974 0.955/0.920/0.979 0.963/0.934/0.983 0.971/0.948/0.986 0.973/0.952/0.987

--- S3: union configs (recall all / India / US | avg cands/S1) ---
base(addr.002,name.005,c4.003 @15/15/10)     0.8845 / 0.8285 / 0.9225 | 33.7
loose@15                                     0.9479 / 0.9210 / 0.9661 | 37.6
loose@25                                     0.9589 / 0.9353 / 0.9748 | 62.9
loose+comb@25                                0.9726 / 0.9513 / 0.9870 | 75.1
comb.1+addr.02+c4@25                         0.9726 / 0.9510 / 0.9871 | 62.0
loose+comb@40                                0.9804 / 0.9675 / 0.9892 | 127.1

--- S3: miss reasons for widest union (176 missed pairs) ---
India  foreign_script_name                    51  (29.0%)
India  shares_addr_token(rank/crowded)        46  (26.1%)
US     shares_addr_token(rank/crowded)        37  (21.0%)
India  empty_addr                             21  (11.9%)
US     empty_addr                             16  (9.1%)
US     shares_number_only                      2  (1.1%)
US     nothing_shared                          2  (1.1%)
US     foreign_script_name                     1  (0.6%)

examples (S1 name | S1 addr || missed name | missed addr):
  [India / foreign_script_name]
     Life Foods Limited | Door No:31, M-Cross, New Gruppan Palaya D.R.C Post, Bangalore, Karnataka  ||  ಲೈಫ್ ಫುಡ್ಸ್ ಲಿಮಿಟೆಡ್ | Dor No:31, Bengaluru, Bangalore, KA
     Digital Energy Private Limited | Plot No 204-B, Venkat Damji Residency, Flat No 401 Near B.K. Guda Park, Balkampet, Hyderabad, Telangana  ||  డిజిటల్ ఎనర్జీ ప్రైవేట్ లిమిటెడ్ | H.no 204-B, NULL, Hyderabad, తెలంగాణ
     Modern Jain Industries Limited | Khasra No 33/24/1, Etc, Gagan Nagar Giaspura, Ludhiana, Punjab  ||  ਮਾਡਰਨ ਜੈਨ ਇੰਡਸਟ੍ਰੀਜ਼ ਲਿਮਟਿਡ | 32/24/1, Ludhiana, PB
  [India / shares_addr_token(rank/crowded)]
     Apex & Sons Private Limited | Bangalore South, 2Nd Floor, 1St Main, Sector-6, Hsr Layout, Bangalore, No. 574/A, Karnataka  ||  Mr apexsons.com | H.no 574/A, Bangalore, Bangalore South, KA
     Vision Constructions LLP | B.No.61, Indraprasth Bung. Nr. Binori Ghuma, Ta. Dascroi, Ahmedabad, Gujarat  ||  Center Vision LLP | B.no.61, Kathwada, ગુજરાત
     Agrawal Crane Co | No.528, 8Th Cross, 1St Phase, Mcechs Layout, Bangalore North, Bangalore, Karnataka  ||  Agrawal Care Co | No.00528, Bangalore North, Bangalore, ಕರ್ನಾಟಕ
  [US / shares_addr_token(rank/crowded)]
     Corner Nails Inc | 4 Main Street, Crawford, NY  ||  Halosyn | 4 Main St, Pine Bush, New York
     Merci Hamilton Obic Co | 2677 47, Hopewell, NY  ||  Noviarcaria | 677 47, Canandaigua, New York
     Fresh Bike Shop | 810 Florida Avenue, WI, Village Of Little Chute  ||  @freshbike | 0810 Florida Ave, Village Of Little Chute, Wisconsin
  [India / empty_addr]
     White Software Limited | Shop No.1804/1, Maulana Azad Marg, Luhar Chowk, Kutch, Mandvi, Kachchh, Gujarat  ||  White Limited Center |
     Pune Overseas | Sn - 17 - 6A - 6B - 7, Five Garden C - 101, Row Hos, Rahatni Chd, Pune, Maharashtra  ||  Pune-0verseas |
     Super Healthcare Private Limited | Grand Majestic Shop No 1 -31 1St Floor 6Th Main Gandhi Nagar Opp Gubbi Veerana Rangamand, Hira, Bangalore, Karnataka  ||  Super Super Hetaslthaere Private [Limited] |
  [US / empty_addr]
     United Committee | 18 Derby Lane, Lillington, NC  ||  United [Center] |
     Federal Veterinary Clinic | 5306 Yaupon Drive, Arlington, TX  ||  Federa1 Veterinary |
     Williams Federal LLC | 717 28th Court, Des Moines, IA  ||  Williams LLC Partners |
  [US / shares_number_only]
     Dental Liberty Associates LLC | 25111 21st Avenue, Unit 22, Phoenix, AZ  ||  Dental Liberty Associates LLC District | 5111 Twenty First Ave, Arizona, # 22, Phoenxi
     Anchor Inc | MN, Unit Unit 108, 669 D, Vadnais Heights  ||  Anchor Inc | D, # Unit 108, Saint PAUL, Minnesota

Done.
```

```
python code\business_entity_resolution\src\blocking.py --n 5000
```

```
loading train files ...
loaded in 26s
S1 subsample: 5,000
S1 names: 100%|███████████████████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 277709.62row/s]
S1 addrs: 100%|███████████████████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 142688.64row/s]
S2 names: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:17<00:00, 294202.70row/s]
S2 addrs: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:36<00:00, 137718.22row/s]
S3 names: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:16<00:00, 312025.74row/s]
S3 addrs: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:27<00:00, 189035.57row/s]

=== blocking vs S2 ===

  [S2 | India] queries=2,041 targets=2,017,799
      S2/India/comb: fit 13s vocab=275,523 nnz/row=10.0
      S2/India/comb: peak nnz/chunk 181,073,952
      S2/India/addr: fit 11s vocab=232,325 nnz/row=5.5
      S2/India/addr: peak nnz/chunk 22,156,700
      S2/India/glue: fit 11s vocab=189,674 nnz/row=9.1
      S2/India/glue: peak nnz/chunk 17,640,320
  [S2 | India] done 128s

  [S2 | US] queries=2,959 targets=3,016,817
      S2/US/comb: fit 15s vocab=238,294 nnz/row=7.6
      S2/US/comb: peak nnz/chunk 141,118,091
      S2/US/addr: fit 10s vocab=161,625 nnz/row=3.7
      S2/US/addr: peak nnz/chunk 13,584,050
      S2/US/glue: fit 23s vocab=257,242 nnz/row=14.4
      S2/US/glue: peak nnz/chunk 36,653,333
  [S2 | US] done 124s

S2: pair recall 0.9849 | S1 fully recalled 0.9741 | avg candidates/S1 67.5 | 255s
   India: pair recall 0.9721 | full 0.9523 | avg cands 65.4
   US: pair recall 0.9939 | full 0.9888 | avg cands 69.0
   per-channel recall (alone) / avg cands:
      comb   0.9711 / 25.0
      addr   0.8857 / 25.0
      glue   0.6851 / 25.0
      hn     0.7752 / 9.3

=== blocking vs S3 ===

  [S3 | India] queries=2,041 targets=2,115,547
      S3/India/comb: fit 13s vocab=258,676 nnz/row=9.6
      S3/India/comb: peak nnz/chunk 170,454,889
      S3/India/addr: fit 12s vocab=208,376 nnz/row=5.2
      S3/India/addr: peak nnz/chunk 27,352,720
      S3/India/glue: fit 12s vocab=194,417 nnz/row=10.1
      S3/India/glue: peak nnz/chunk 18,689,591
  [S3 | India] done 118s

  [S3 | US] queries=2,959 targets=3,170,056
      S3/US/comb: fit 16s vocab=248,505 nnz/row=7.9
      S3/US/comb: peak nnz/chunk 91,636,718
      S3/US/addr: fit 10s vocab=163,901 nnz/row=3.8
      S3/US/addr: peak nnz/chunk 13,580,238
      S3/US/glue: fit 24s vocab=259,819 nnz/row=14.8
      S3/US/glue: peak nnz/chunk 38,818,666
  [S3 | US] done 129s

S3: pair recall 0.9790 | S1 fully recalled 0.9608 | avg candidates/S1 67.5 | 250s
   India: pair recall 0.9593 | full 0.9247 | avg cands 66.6
   US: pair recall 0.9924 | full 0.9851 | avg cands 68.1
   per-channel recall (alone) / avg cands:
      comb   0.9633 / 25.0
      addr   0.8490 / 25.0
      glue   0.6769 / 25.0
      hn     0.7329 / 8.8
```

```
python code\business_entity_resolution\src\features.py --n 3000
```

```
loading ...
S1 names: 100%|███████████████████████████████████████████████████████████████| 3000/3000 [00:00<00:00, 349408.86row/s]
S1 addrs: 100%|███████████████████████████████████████████████████████████████| 3000/3000 [00:00<00:00, 150107.51row/s]
S2 names: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:12<00:00, 409946.07row/s]
S2 addrs: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:25<00:00, 196437.30row/s]
  [S2] channels:   0%|                                                                         | 0/8 [00:00<?, ?chan/s]
  [S2 | India] queries=1,226 targets=2,017,799
      S2/India/comb: fitting on 2,017,799 rows ...
      S2/India/comb: fit 12s vocab=275,523 nnz/row=10.0
                                                                                                                             S2/India/comb: peak nnz/chunk 180,029,831
  [S2] channels:  12%|████████▏                                                        | 1/8 [00:23<02:44, 23.48s/chan]      S2/India/addr: fitting on 1,959,953 rows ...
      S2/India/addr: fit 10s vocab=232,325 nnz/row=5.5
                                                                                                                             S2/India/addr: peak nnz/chunk 22,156,700
  [S2] channels:  25%|████████████████▎                                                | 2/8 [00:36<01:42, 17.14s/chan]      S2/India/glue: fitting on 2,017,799 rows ...
      S2/India/glue: fit 10s vocab=189,674 nnz/row=9.1
                                                                                                                             S2/India/glue: peak nnz/chunk 17,640,320
  [S2] channels:  50%|████████████████████████████████▌                                | 4/8 [01:34<01:49, 27.30s/chan]  [S2 | India] done 94s

  [S2 | US] queries=1,774 targets=3,016,817
      S2/US/comb: fitting on 3,016,817 rows ...
      S2/US/comb: fit 14s vocab=238,294 nnz/row=7.6
                                                                                                                             S2/US/comb: peak nnz/chunk 141,118,091
  [S2] channels:  62%|████████████████████████████████████████▋                        | 5/8 [02:04<01:24, 28.12s/chan]      S2/US/addr: fitting on 2,905,696 rows ...
      S2/US/addr: fit 9s vocab=161,625 nnz/row=3.7
                                                                                                                             S2/US/addr: peak nnz/chunk 13,584,050
  [S2] channels:  75%|████████████████████████████████████████████████▊                | 6/8 [02:16<00:45, 22.55s/chan]      S2/US/glue: fitting on 3,016,817 rows ...
      S2/US/glue: fit 20s vocab=257,242 nnz/row=14.4
                                                                                                                             S2/US/glue: peak nnz/chunk 36,653,333
  [S2] channels: 100%|█████████████████████████████████████████████████████████████████| 8/8 [03:19<00:00, 28.02s/chan]  [S2 | US] done 104s
  [S2] channels: 100%|█████████████████████████████████████████████████████████████████| 8/8 [03:19<00:00, 24.97s/chan]
expand pairs: 100%|█████████████████████████████████████████████████████████████| 3000/3000 [00:00<00:00, 12972.82s1/s]
  building pair table (202,465 pairs) ...
pairs: 202,465
features x14: 100%|██████████████████████████████████████████████████████| 202465/202465 [00:01<00:00, 144697.42pair/s]
  assembling feature frame ...
  features: 202,465 pairs in 2s (97,774 pairs/s)
context features: 100%|████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  5.64step/s]
60 feature columns; positive rate 0.0244
rk_hn          0.863
sc_hn          0.863
lead_num_eq    0.669
legal_eq       0.656
sc_addr        0.630
rk_addr        0.630
sc_glue        0.630
rk_glue        0.630
sc_comb        0.630
rk_comb        0.630
legal_jac      0.185
n_tset         0.099
```

```
python code\business_entity_resolution\src\build_train_pairs.py --n 100000
```

```
loading train files ...
S1 subsample: 100,000
S1 names: 100%|███████████████████████████████████████████████████████████| 100000/100000 [00:00<00:00, 265218.88row/s]
S1 addrs: 100%|███████████████████████████████████████████████████████████| 100000/100000 [00:00<00:00, 140999.18row/s]
S2 names: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:18<00:00, 276268.10row/s]
S2 addrs: 100%|█████████████████████████████████████████████████████████| 5034616/5034616 [00:33<00:00, 150185.70row/s]
S3 names: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:14<00:00, 375065.05row/s]
S3 addrs: 100%|█████████████████████████████████████████████████████████| 5285603/5285603 [00:28<00:00, 186935.15row/s]

=== S2: blocking ===
  [S2] channels:   0%|                                                                         | 0/8 [00:00<?, ?chan/s]
  [S2 | India] queries=40,138 targets=2,017,799
      S2/India/comb: fitting on 2,017,799 rows ...
      S2/India/comb: fit 13s vocab=275,523 nnz/row=10.0
                                                                                                                             S2/India/comb: peak nnz/chunk 187,679,431
  [S2] channels:  12%|████████                                                        | 1/8 [05:41<39:48, 341.26s/chan]      S2/India/addr: fitting on 1,959,953 rows ...
      S2/India/addr: fit 10s vocab=232,325 nnz/row=5.5
                                                                                                                             S2/India/addr: peak nnz/chunk 22,156,700
  [S2] channels:  25%|████████████████                                                | 2/8 [06:36<17:19, 173.26s/chan]      S2/India/glue: fitting on 2,017,799 rows ...
      S2/India/glue: fit 10s vocab=189,674 nnz/row=9.1
                                                                                                                             S2/India/glue: peak nnz/chunk 18,048,250
  [S2] channels:  50%|████████████████████████████████▌                                | 4/8 [08:32<06:19, 94.98s/chan]  [S2 | India] done 511s

  [S2 | US] queries=59,862 targets=3,016,817
      S2/US/comb: fitting on 3,016,817 rows ...
      S2/US/comb: fit 13s vocab=238,294 nnz/row=7.6
                                                                                                                             S2/US/comb: peak nnz/chunk 145,328,790
  [S2] channels:  62%|████████████████████████████████████████                        | 5/8 [18:04<13:21, 267.10s/chan]      S2/US/addr: fitting on 2,905,696 rows ...
      S2/US/addr: fit 13s vocab=161,625 nnz/row=3.7
                                                                                                                             S2/US/addr: peak nnz/chunk 14,119,513
  [S2] channels:  75%|████████████████████████████████████████████████                | 6/8 [19:36<06:55, 207.50s/chan]      S2/US/glue: fitting on 3,016,817 rows ...
      S2/US/glue: fit 30s vocab=257,242 nnz/row=14.4
                                                                                                                             S2/US/glue: peak nnz/chunk 38,482,749
  [S2] channels: 100%|████████████████████████████████████████████████████████████████| 8/8 [26:08<00:00, 188.28s/chan]  [S2 | US] done 1054s
  [S2] channels: 100%|████████████████████████████████████████████████████████████████| 8/8 [26:08<00:00, 196.09s/chan]
=== S2: pairs + features ===
expand pairs: 100%|██████████████████████████████████████████████████████████| 100000/100000 [00:10<00:00, 9328.06s1/s]
  building pair table (6,759,037 pairs) ...
features x14: 100%|████████████████████████████████████████████████████| 6759037/6759037 [00:46<00:00, 146373.52pair/s]
  assembling feature frame ...
  features: 6,759,037 pairs in 77s (87,625 pairs/s)

=== S3: blocking ===
  [S3] channels:   0%|                                                                         | 0/8 [00:00<?, ?chan/s]
  [S3 | India] queries=40,138 targets=2,115,547
      S3/India/comb: fitting on 2,115,547 rows ...
      S3/India/comb: fit 17s vocab=258,676 nnz/row=9.6
                                                                                                                             S3/India/comb: peak nnz/chunk 178,132,920
  [S3] channels:  12%|████████                                                        | 1/8 [06:10<43:15, 370.79s/chan]      S3/India/addr: fitting on 2,050,583 rows ...
      S3/India/addr: fit 9s vocab=208,376 nnz/row=5.2
                                                                                                                             S3/India/addr: peak nnz/chunk 27,658,766
  [S3] channels:  25%|████████████████                                                | 2/8 [07:26<19:42, 197.02s/chan]      S3/India/glue: fitting on 2,115,547 rows ...
      S3/India/glue: fit 11s vocab=194,417 nnz/row=10.1
                                                                                                                             S3/India/glue: peak nnz/chunk 19,085,887
  [S3] channels:  50%|████████████████████████████████                                | 4/8 [09:28<06:57, 104.40s/chan]  [S3 | India] done 559s

  [S3 | US] queries=59,862 targets=3,170,056
      S3/US/comb: fitting on 3,170,056 rows ...
      S3/US/comb: fit 15s vocab=248,505 nnz/row=7.9
                                                                                                                             S3/US/comb: peak nnz/chunk 95,631,851
  [S3] channels:  62%|████████████████████████████████████████                        | 5/8 [14:38<08:56, 178.76s/chan]      S3/US/addr: fitting on 3,059,088 rows ...
      S3/US/addr: fit 9s vocab=163,901 nnz/row=3.8
                                                                                                                             S3/US/addr: peak nnz/chunk 13,877,032
  [S3] channels:  75%|████████████████████████████████████████████████                | 6/8 [15:44<04:40, 140.32s/chan]      S3/US/glue: fitting on 3,170,056 rows ...
      S3/US/glue: fit 22s vocab=259,819 nnz/row=14.8
                                                                                                                             S3/US/glue: peak nnz/chunk 41,133,251
  [S3] channels: 100%|████████████████████████████████████████████████████████████████| 8/8 [20:40<00:00, 134.87s/chan]  [S3 | US] done 670s
  [S3] channels: 100%|████████████████████████████████████████████████████████████████| 8/8 [20:40<00:00, 155.02s/chan]
=== S3: pairs + features ===
expand pairs: 100%|█████████████████████████████████████████████████████████| 100000/100000 [00:06<00:00, 14568.79s1/s]
  building pair table (6,752,345 pairs) ...
features x14: 100%|████████████████████████████████████████████████████| 6752345/6752345 [00:43<00:00, 155391.22pair/s]
  assembling feature frame ...
  features: 6,752,345 pairs in 67s (100,752 pairs/s)

total pairs: 13,511,382
context features: 100%|████████████████████████████████████████████████████████████████| 6/6 [02:40<00:00, 26.81s/step]
positive rate 0.0251  (339,509 positives)
saved logs/pairs_train.parquet in 3423s total
```

```
python code\business_entity_resolution\src\train_model.py
```

```
pairs 13,511,382 | S1 100,000 | positives 339,509
features 60 | train pairs 10,806,534 | val pairs 2,704,848
lightgbm:  68%|█████████████████████████████▉              | 1362/2000 [03:55<01:50,  5.77rnd/s, binary_logloss=0.0064]
trained in 236s, best iteration 1302
pair-level  AUC 0.99959 | AP 0.99011
```

```
python code\business_entity_resolution\src\train_model.py --skip-train
```

```
pairs 13,511,382 | S1 100,000 | positives 339,509
features 60 | train pairs 10,806,534 | val pairs 2,704,848
loaded model from logs/lgbm.txt
pair-level  AUC 0.99959 | AP 0.99011

threshold sweep (macro F0.5 on validation S1, singletons included):
   thr     1to1    plain
  0.20   0.9184   0.9183
  0.25   0.9271   0.9271
  0.30   0.9340   0.9340
  0.35   0.9383   0.9383
  0.40   0.9421   0.9422
  0.45   0.9453   0.9453
  0.50   0.9474   0.9474
  0.55   0.9484   0.9484
  0.60   0.9498   0.9498
  0.65   0.9501   0.9501
  0.70   0.9497   0.9497
  0.75   0.9486   0.9486
  0.80   0.9473   0.9473
  0.85   0.9448   0.9448
  0.90   0.9394   0.9394
  0.95   0.9257   0.9257

BEST macro F0.5 (1-to-1) = 0.9501 at threshold 0.68
  India: 0.9213  (n=8,027)
  US: 0.9694  (n=11,972)
  singletons: 1070/1150 correctly empty
  mean predicted matches/S1 3.19 | mean true 3.46

top 25 features by gain:
base            0.3514
rk_comb         0.2911
hn_jac          0.0594
comb_score      0.0464
sc_comb         0.0419
gap_second_o    0.0211
legal_jac       0.0126
n_lev           0.0123
hn_conflict     0.0117
a_tset          0.0110
n_channels      0.0103
rank_s1         0.0096
n_len_b         0.0072
n_comp_jw       0.0070
n_tset          0.0066
best_rank       0.0064
n_comp_part     0.0061
rk_glue         0.0055
legal_eq        0.0049
n_jac           0.0044
n_jw            0.0040
a_cov_b         0.0038
a_tsort         0.0037
n_part          0.0036
a_part          0.0035

FALSE POSITIVES (wrong merges): 1,142 total
    p                                   name_a                                                                    addr_a                                name_b                                                         addr_b
0.910      Ernakulam Hospitals Private Limited               H7/7Th Floor, Heavenly Plaza, Vazhakkala, Ernakulam, Kerala       Ernakulam Tech Private  Limited                 FLAT NO : E7 HEAVENLY PLAZA, ERNAKULAM, Kerala
0.825                                Titan Inc                                                1407 Main Street, Hays, KS                            Titan Inc.                           1407 Choctaw St, Tahlequah, Oklahoma
0.816        Office & Brothers Private Limited          Lig 506 Aurangabad, Hsg Board C I D C O, Aurangabad, Maharashtra                              Deltanex        Lig 506 Aurangabad, Hsg Board C I D C O, Aurangabad, MH
0.730                            Cape Ventures                            444/4 Flat No. 2, Raja Park, Jaipur, Rajasthan                 Pvt Cape Ventures Ltd                      H.NO 444/13 FLAT NO. 2, JAIPUR, Rajasthan
0.681              Elite Payroll Networks PLLC                                            2005 Dayton Drive, Melissa, TX                Elite Payroll Nétworks
0.902        Nashik Realestate Private Limited           Flat No-08 Pl.No-10, 11, Ojas Heights Appt, Nashik, Maharashtra        ... Nashik Weaving Private Ltd                       FLAT NO-08 PL.NO-10, NASHIK, Maharashtra
0.962                 Buffalo Community Church                          NY, 628 Fulton Street, Unit Apartment 1, Buffalo                     Buffalo Community                                      SMITH STREET, NY, BUFFALO
0.836                    Neal Automotive Group                                      2419 Parker Boulevard, Tonawanda, NY                  Corp Neal Automotive                            426 PARKER BOULEVARD, TONAWANDA, NY
0.834        Gulf Great Commercial Corporation                              OH, 1954 Big Run Bluff Boulevard, Grove City           GULF GREAT COMMERCIAL CORP.                      001967 BIG RUN BLUFF BLVD, GROVE CITY, OH
0.722                              Staten Sony                                  1455 Arthur Burch Drive, Bourbonnais, IL                     (Ltd) Staten Sony                 1459 Arthur Burch Drive, Bourbonnais, Illinois
0.795        International Biomedical Networks                             49825 Mulligan Creek Road, Town Of Barnes, WI International Biomedical Networks Ltd           49832 Mulligan Creek Road, Town Of Barnes, Wisconsin
0.943 Malappuram Realstructure Private Limited Building No. 38/2 Pandikadavath, Karathode, Malappuram, Kerala, Malapuram            Malappuram Enviro Pvt  Ltd No. 38/2, Pandikadavath, Karathode, Tirurangadi, Malapuram, KL

FALSE NEGATIVES (missed matches): 5,255 total
    p                               name_a                                                                                                                    addr_a                               name_b                                                                                                          addr_b
0.039                                Ember                                                                                       1560 Ellie Avenue, Fayetteville, NC                        EMBER SERVICE                                                                                NC, 1561 ELLIE AVE, FAYETTEVILLE
0.520  Sarvatman Securities Public Limited                                             Flat No. 105, 1St Floor, H No. 9/2, East Patel Nagar, Delhi, New Delhi, Delhi        Sarvatman Securities [Public]                                             DOOR NO 1-05 , 1ST FLOOR, H NO. 9/2, EAST PATEL NAGAR, DELHI, Delhi
0.672                    Unique Consulting                                      Flat No. 101, Saidhan Apartment, Near D-Mart Mall, Kphb Colony, Hyderabad, Telangana                   యూనిక్ కన్సల్టింగ్                                    Door No 101, Saidhan Apartment, Near D-mart Mall, Kphb Colony, Hyderabad, TG
0.114                      Apex Investment                                                    C/O Md Ali, At New, Azimabad Colony, Gaighat, Sampatchak, Patna, Bihar                  एपेक्स इन्वेस्टमेंट                                                                                        C/O MD ALI, PATNA, Bihar
0.079 Neotech Institute of Technology Corp               1220, Shilp Epitome, Off Rajpath Rangoli Road, Nr Infostretch, Bodakdev, Ahmadabad City, Ahmedabad, Gujarat Neotech Isntftuhe of Technology Corp                                       
0.045             Black It Private Limited                                                      Rajdeep Society-1, Dr. K. K. Pambhar Street, Rajkot, Gujarat, Bl-106          બ્લેક આઈટી પ્રાઇવેટ લિમિટેડ                                                                                 PLOT 817 BL-106, RAJKOT, ગુજરાત
0.348         Regional Payment Products LP                                                                                     4833 9th Street, Arlington County, VA            Regional Payment Products                                                                                 833 9th St, Arlington, Virginia
0.474                  Black International                                            Shop No 05 Plot No W57/58, Shahabaj Gaon Thane Navi Mumbai, Thane, Maharashtra                      ब्लैक इंटरनेशनल                                                                                    05 Plot No W57/58, Thane, MH
0.360              Henderson and Weber LLC                                                                                             175 Basalt Drive, Maxwell, TX                  Henderson and Weber                                                                                 75 Basalt Drive, Maxwell, Texas
0.332          Dharshini Promoters Limited                                                                                  D 368, D Block, Ghaziabad, Uttar Pradesh                  Dharshini Promoters                                                                                 D 3-68, Ghaziabad, उत्तर प्रदेश
0.287              Correy Blue Patriot LLC                                                                                 8756 Highland Road, Town Of Gibraltar, WI                          Correy Blue                                                                           Highland Rd, FISH CREK CDP, Wisconsin
0.037                     Classic Plastics 324, Narmada Apts, Behind Oshiwara Police Station, New Link Road, Jogeshwari (West), Mumbai, Bandra Suburban, Maharashtra                Classic Center #60774 24, Narmada Apts, Behind Oshiwara Police Station, New Link Road, Jogeshwari (West), Mumbai, Bandra Suburban, MH
```