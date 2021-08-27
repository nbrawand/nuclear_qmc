# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/li_6_exp/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 3,
    "n_proton": 3,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "n_blocks": 4,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 4,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 40,
        "print_local_energy": true,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.0,
        "include_3body": false,
        "model_string": "o"
    },
    "wave_function": {
        "add_partition_jastro": true,
        "coefficients": [
            0.3333,
            0.3333,
            0.3333
        ],
        "jastro_list": [
            "2b",
            "3b",
            "sigma",
            "tau",
            "sigma_tau"
        ],
        "n_dense": 4,
        "n_hidden_layers": 1,
        "orbitals": [
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y11_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y11_d_p"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y10_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y10_d_p"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y1m1_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y1m1_d_p"
            ]
        ],
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
## Wave Function Parameters
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/li_6_exp/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 35.374035172131336 | 1.549476420581159
optimization step | 1 | 28.113466798251448 | 3.767478485354725
optimization step | 2 | 24.805777048854946 | 2.3920091607141756
optimization step | 3 | 18.00570104742515 | 1.8105977377598967
optimization step | 4 | 21.174100964909556 | 1.3440541625765419
optimization step | 5 | 13.607036220671956 | 1.6628715491158026
optimization step | 6 | 10.90847928389563 | 2.5071684929866302
optimization step | 7 | 12.573459519766729 | 1.9259932067648509
optimization step | 8 | 9.877110025093028 | 0.7888760483536765
optimization step | 9 | 7.267258134375624 | 0.3931692066339986
optimization step | 10 | 3.713685667083346 | 1.0477472722465653
optimization step | 11 | 4.8580412140392335 | 1.2056619619553128
optimization step | 12 | 1.6577138722894849 | 0.9446894493505176
optimization step | 13 | -0.743930325196322 | 1.2376996775629971
optimization step | 14 | -2.331602394331516 | 2.1607840593572876
optimization step | 15 | -2.384905994945703 | 1.3632900435153463
optimization step | 16 | -4.50373931231408 | 1.3262636037741358
optimization step | 17 | -5.137483950336495 | 1.0240870281807923
optimization step | 18 | -5.257818684063525 | 1.224973810338748
optimization step | 19 | -4.517412374729472 | 0.7212133387340514
optimization step | 20 | -5.389773739863715 | 0.8039182024840498
optimization step | 21 | -10.011784216460178 | 1.0326785812243475
optimization step | 22 | -8.1282935032224 | 0.8335985284337987
optimization step | 23 | -8.966253344152335 | 1.3714873647378123
optimization step | 24 | -13.310805171252426 | 1.3459139419783066
optimization step | 25 | -11.14792937016637 | 1.4010568152605671
optimization step | 26 | -12.04316319456912 | 1.7043824116523383
optimization step | 27 | -15.116583807731452 | 1.1729688549498343
optimization step | 28 | -17.06504802820582 | 1.0506612025089044
optimization step | 29 | -15.893913391992117 | 1.4820103460533751
optimization step | 30 | -17.99474464279753 | 0.8366667257591104
optimization step | 31 | -17.813605122700526 | 1.0158287743707055
optimization step | 32 | -20.70510131635914 | 2.375705253013423
optimization step | 33 | -18.301881411020837 | 1.3299628185721668
optimization step | 34 | -18.023049192354858 | 1.1061988303547068
optimization step | 35 | -21.541348980074496 | 0.6381707315779779
optimization step | 36 | -20.466246021382226 | 0.986404930218828
optimization step | 37 | -21.264755006705983 | 1.0359742618334764
optimization step | 38 | -22.075138935668164 | 0.6118157208278585
optimization step | 39 | -24.81849892577378 | 0.6309970780646227
optimization step | 40 | -23.296089515456053 | 0.34052675236496965
optimization step | 41 | -25.662723300521996 | 0.5656820284007388
optimization step | 42 | -26.20002253431893 | 0.8212493132158097
optimization step | 43 | -27.15953497135143 | 0.31380974692422864
optimization step | 44 | -27.306572957097387 | 0.3808217062600353
optimization step | 45 | -28.769738501081935 | 0.8064368977709968
optimization step | 46 | -26.637760920714 | 0.7658770865903439
optimization step | 47 | -31.626658535496425 | 0.839501007579374
optimization step | 48 | -27.8941421868884 | 0.7230362241042957
optimization step | 49 | -31.241935969658464 | 0.7512511209677905
optimization step | 50 | -30.65384890587086 | 1.083788112295628
optimization step | 51 | -31.16567466388156 | 1.5970586744741824
optimization step | 52 | -31.480907977943197 | 1.2012110978198787
optimization step | 53 | -29.969521253052886 | 1.1043551398216571
optimization step | 54 | -34.19115679004015 | 0.9977724496345174
optimization step | 55 | -33.371688530468894 | 0.8942109055589823
optimization step | 56 | -35.75728625203904 | 0.7330213896556818
optimization step | 57 | -33.51462299125359 | 0.572569251212412
optimization step | 58 | -34.42273292230886 | 0.5001456160415907
optimization step | 59 | -34.71701876752873 | 0.48228499591489615
optimization step | 60 | -36.09666834810913 | 0.9750245863457151
optimization step | 61 | -33.90306131951991 | 0.8664731257872901
optimization step | 62 | -36.128011997734525 | 0.9569550881552777
optimization step | 63 | -37.85255518674102 | 0.252911947176731
optimization step | 64 | -38.17279513652587 | 0.19711611229449955
optimization step | 65 | -37.57674731472491 | 0.9619458995933126
optimization step | 66 | -38.061953332135076 | 0.5081641710163656
optimization step | 67 | -38.54995112157349 | 0.6703332456212316
optimization step | 68 | -37.72148736110918 | 0.9673671083335184
optimization step | 69 | -38.489694554251884 | 0.44868050369718104
optimization step | 70 | -37.90886114977587 | 0.43430343235897423
optimization step | 71 | -38.492116765373694 | 0.5308491057334467
optimization step | 72 | -39.50119783348232 | 1.1104463321629738
optimization step | 73 | -38.971581749239164 | 0.6467200328109551
optimization step | 74 | -38.936886839319655 | 0.9205070223048964
optimization step | 75 | -38.76929778504668 | 0.4643178910109073
optimization step | 76 | -39.58531641949576 | 0.7189153814850238
optimization step | 77 | -39.837145031443484 | 0.7757417629506499
optimization step | 78 | -40.057298254942054 | 0.5509629500409626
optimization step | 79 | -39.79863471835415 | 0.6549795431570002
optimization step | 80 | -40.02706341218577 | 0.5698210504952381
optimization step | 81 | -40.66131317640384 | 1.0962335752528973
optimization step | 82 | -42.13062273486784 | 0.9655544330850492
optimization step | 83 | -39.47080463680504 | 0.8820635826149519
optimization step | 84 | -38.29938214169536 | 1.0488772017821806
optimization step | 85 | -38.752744263634824 | 0.7154718976717598
optimization step | 86 | -39.56784434918862 | 0.35075107648903886
optimization step | 87 | -40.65404796645096 | 0.3773258040730031
optimization step | 88 | -40.52713600711733 | 0.5807067471560364
optimization step | 89 | -41.165519657740624 | 0.3484019698777719
optimization step | 90 | -40.32177292096743 | 0.36937756790614035
optimization step | 91 | -40.80638964976003 | 1.094035709443289
optimization step | 92 | -40.57629179090931 | 0.4940377290519882
optimization step | 93 | -40.49431491605339 | 0.421905749758064
optimization step | 94 | -40.66734199764987 | 0.24061424013667623
optimization step | 95 | -40.337347267080276 | 0.7557752326680861
optimization step | 96 | -40.61833142610291 | 0.9024645218217752
optimization step | 97 | -42.40303834974907 | 0.4253205351709481
optimization step | 98 | -40.05164969581967 | 0.76324216036555
optimization step | 99 | -40.976774304799235 | 0.6311250161072375
optimization step | 100 | -41.32810633420341 | 0.5740744362836698
optimization step | 101 | -42.009462585175825 | 0.36607516426607034
optimization step | 102 | -41.43905303309826 | 0.517287011688776
optimization step | 103 | -41.03024128544295 | 0.9738049574749117
optimization step | 104 | -41.87756774873052 | 0.6559177714646531
optimization step | 105 | -40.89875625195978 | 0.5763521498886804
optimization step | 106 | -40.126775662241414 | 0.6569222420544509
optimization step | 107 | -42.26855574909845 | 0.432674755641446
optimization step | 108 | -41.50153517454804 | 0.8114812707963246
optimization step | 109 | -41.150137439731054 | 0.31496016364579266
optimization step | 110 | -41.673790974449744 | 0.43589122281217696
optimization step | 111 | -42.80768665037586 | 0.7296355047937089
optimization step | 112 | -42.637267629435534 | 0.7696666520923194
optimization step | 113 | -41.64522050083689 | 0.7716113320653922
optimization step | 114 | -40.64019344679831 | 0.466635457281647
optimization step | 115 | -41.7660877091304 | 0.20617402625032283
optimization step | 116 | -42.21147594730817 | 0.5144146848963806
optimization step | 117 | -40.96627041464882 | 0.5877542998552834
optimization step | 118 | -41.76557833924784 | 0.7495088475618275
optimization step | 119 | -42.679186482639295 | 0.7181278007550094
optimization step | 120 | -41.25364676547439 | 0.41041108617450983
optimization step | 121 | -41.734869180533735 | 0.2825161414213637
optimization step | 122 | -41.65421336369135 | 0.3027193583195107
optimization step | 123 | -41.52102340233287 | 0.4596868228565245
optimization step | 124 | -42.158415851769526 | 0.4605300414547022
optimization step | 125 | -41.73540344673905 | 0.4252285486647253
optimization step | 126 | -40.913399280733444 | 0.840042003713431
optimization step | 127 | -40.83448381800475 | 0.39454606738094816
optimization step | 128 | -41.81300068087692 | 0.3926295972247027
optimization step | 129 | -42.3654914518399 | 0.5168012391119151
optimization step | 130 | -41.2036641575177 | 0.5425694221419894
optimization step | 131 | -41.428397773081755 | 0.6065846692069412
optimization step | 132 | -41.19823315218199 | 0.4217427391734015
optimization step | 133 | -41.416924267319864 | 0.8486241992337528
optimization step | 134 | -41.35470226928715 | 0.2211046852244852
optimization step | 135 | -42.556674294782844 | 0.30413260793044955
optimization step | 136 | -41.36878803106016 | 0.1312424573896777
optimization step | 137 | -41.68299967781599 | 0.2424756051381547
optimization step | 138 | -42.592736061917535 | 0.30585984695158935
optimization step | 139 | -41.84497250967718 | 0.7348106501842084
optimization step | 140 | -42.39178640813059 | 0.3902137989203119
optimization step | 141 | -42.794213466183095 | 0.524153746831823
optimization step | 142 | -43.256292673654514 | 0.4299599065176396
optimization step | 143 | -41.29154819726216 | 0.5052583067846133
optimization step | 144 | -42.0981438493518 | 0.4663425223441139
optimization step | 145 | -42.78047780212387 | 0.7709620665818693
optimization step | 146 | -42.62123518351146 | 0.3617706699083205
optimization step | 147 | -41.503086411970735 | 0.7163479596360758
optimization step | 148 | -41.695030536543825 | 0.2141995866978684
optimization step | 149 | -41.88679221462387 | 0.49410035583488265
optimization step | 150 | -42.38708608855708 | 0.18228859077552026
optimization step | 151 | -41.312485188848214 | 0.16884242201834598
optimization step | 152 | -41.479708428680716 | 0.3465702198758943
optimization step | 153 | -41.97757808064665 | 0.2689154779026357
optimization step | 154 | -41.67975341010446 | 0.4123140943152362
optimization step | 155 | -41.89936796696068 | 0.519423200682799
optimization step | 156 | -40.885422769848475 | 0.06444390051189988
optimization step | 157 | -42.10511321469302 | 0.33567600652928503
optimization step | 158 | -41.04025917422716 | 0.3301407551840546
optimization step | 159 | -42.295426063998256 | 0.4951235935583452
optimization step | 160 | -41.419634175013215 | 0.22238250062527956
optimization step | 161 | -41.44545323437653 | 0.5263733743847177
optimization step | 162 | -42.98671858254374 | 0.43515646667162605
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/li_6_exp/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 3,
    "n_proton": 3,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "n_blocks": 8,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 8,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 50,
        "print_local_energy": true,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.0,
        "include_3body": false,
        "model_string": "o"
    },
    "wave_function": {
        "add_partition_jastro": true,
        "coefficients": [
            0.3333,
            0.3333,
            0.3333
        ],
        "jastro_list": [
            "2b",
            "3b",
            "sigma",
            "tau",
            "sigma_tau"
        ],
        "n_dense": 4,
        "n_hidden_layers": 1,
        "orbitals": [
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y11_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y11_d_p"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y10_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y10_d_p"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y1m1_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y1m1_d_p"
            ]
        ],
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/li_6_exp/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -42.558991054446004 | 0.14976896039219337
optimization step | 1 | -41.88343227130633 | 0.30816475511720304
optimization step | 2 | -42.14607997024667 | 0.2176871829837029
optimization step | 3 | -41.733118918600404 | 0.2783794830048736
optimization step | 4 | -42.17381215555317 | 0.4244655106353981
optimization step | 5 | -41.82978672510795 | 0.29784624547841226
optimization step | 6 | -41.997875659896614 | 0.12953931305316277
optimization step | 7 | -42.20413980664558 | 0.23095400344998754
optimization step | 8 | -41.98852082137501 | 0.3367693464639156
optimization step | 9 | -41.85107987600152 | 0.41772819650485177
optimization step | 10 | -42.228435763856645 | 0.29423307854001574
optimization step | 11 | -41.58086783361149 | 0.3284403820878513
optimization step | 12 | -42.33530761065065 | 0.2877598945517333
optimization step | 13 | -42.21537370285274 | 0.22737509353579768
optimization step | 14 | -42.075311386723584 | 0.3081092466214668
optimization step | 15 | -42.08124552301221 | 0.15831164726168614
optimization step | 16 | -41.99664492028298 | 0.21173430692049347
optimization step | 17 | -42.037304424633305 | 0.19441754817090598
optimization step | 18 | -42.38456442842481 | 0.2342819587775458
optimization step | 19 | -42.13377329238039 | 0.24672567114025337
optimization step | 20 | -42.33225621062994 | 0.2921104224554651
optimization step | 21 | -42.194485176688616 | 0.1960296770205213
optimization step | 22 | -41.72067687692765 | 0.23890094800864034
optimization step | 23 | -42.104551128493036 | 0.3095788075973102
optimization step | 24 | -42.027057957336645 | 0.18195139309888456
optimization step | 25 | -41.95955353030758 | 0.3232378092790127
optimization step | 26 | -42.447445287177665 | 0.18682716300763616
optimization step | 27 | -42.07470666735218 | 0.2761652203297597
optimization step | 28 | -42.36231926295044 | 0.2922503400663181
optimization step | 29 | -41.86385892791131 | 0.3543367472980423
optimization step | 30 | -42.430464577310644 | 0.24562448599949305
optimization step | 31 | -42.24997733311823 | 0.37035172681458584
optimization step | 32 | -41.92766847566364 | 0.2082223284574289
optimization step | 33 | -42.51667970260474 | 0.39064055041695894
optimization step | 34 | -42.005865137203905 | 0.21080598330347675
optimization step | 35 | -42.30184975155923 | 0.3278032279786398
optimization step | 36 | -42.05250618002482 | 0.28562568059789284
optimization step | 37 | -42.1995674754289 | 0.1748952318296526
optimization step | 38 | -41.73297571508268 | 0.24303886452805765
optimization step | 39 | -42.258277324606816 | 0.28373125693517587
optimization step | 40 | -42.34980261225526 | 0.2301348290499225
optimization step | 41 | -42.14162837834743 | 0.3364979529879362
optimization step | 42 | -42.13114032542061 | 0.16705251594260567
optimization step | 43 | -42.09756748635727 | 0.22893803390060136
optimization step | 44 | -41.721903153116955 | 0.14144480494533065
optimization step | 45 | -41.773934039467974 | 0.25993801939196604
optimization step | 46 | -41.94529961815476 | 0.2934129710022835
optimization step | 47 | -42.473449236909566 | 0.1341704385560964
optimization step | 48 | -41.935268775075905 | 0.30781831288665984
optimization step | 49 | -42.15331213395744 | 0.19664433299361728
optimization step | 50 | -42.49929087181967 | 0.13266447746241908
optimization step | 51 | -42.084872788299066 | 0.17658401192724
optimization step | 52 | -42.38327291565831 | 0.18743985839292485
optimization step | 53 | -42.391511897586255 | 0.29665634467748264
optimization step | 54 | -42.1930073919651 | 0.5168232892241746
optimization step | 55 | -41.69825427697533 | 0.22720001221282368
optimization step | 56 | -42.3079188438815 | 0.20080551598673738
optimization step | 57 | -41.967151050471394 | 0.28265632880482816
optimization step | 58 | -42.323028293238444 | 0.1626841605464302
optimization step | 59 | -42.38094547144138 | 0.22115112247965762
optimization step | 60 | -41.19480156925837 | 0.2550537048318537
optimization step | 61 | -42.31248595864252 | 0.1984460137413618
optimization step | 62 | -42.277206359897846 | 0.30723985154201894
optimization step | 63 | -41.8914469118986 | 0.23501656996390877
optimization step | 64 | -41.84830788491892 | 0.321129517406535
optimization step | 65 | -41.94831071697135 | 0.37011382858096564
optimization step | 66 | -42.21600367617121 | 0.16173010605111715
optimization step | 67 | -42.12723319950501 | 0.2822601413177387
optimization step | 68 | -42.0508012717947 | 0.1654618331836113
optimization step | 69 | -42.336811936125066 | 0.13329708828245151
optimization step | 70 | -42.149967497613474 | 0.29259406346608163
optimization step | 71 | -41.635010780973545 | 0.29665731872625933
optimization step | 72 | -42.124967063151644 | 0.2908787313640768
optimization step | 73 | -42.113702623927296 | 0.34674651766327724
optimization step | 74 | -42.17608677134496 | 0.2688695421269144
optimization step | 75 | -42.19461698510629 | 0.3389966390015547
optimization step | 76 | -42.535770999344116 | 0.318538417329692
optimization step | 77 | -41.694813186594345 | 0.3027297182611522
optimization step | 78 | -42.520439712079174 | 0.3145699509027934
optimization step | 79 | -42.28575156194184 | 0.2330875256021995
optimization step | 80 | -41.604339514755736 | 0.2669735760265767
optimization step | 81 | -42.49213403842199 | 0.15584181171082492
optimization step | 82 | -41.419836281938586 | 0.2962864791240083
optimization step | 83 | -42.00455582898648 | 0.2590489337827068
optimization step | 84 | -42.09580820665474 | 0.2733709767978143
optimization step | 85 | -42.17107875425686 | 0.22098428469720757
optimization step | 86 | -42.079234331953394 | 0.15689081289615944
optimization step | 87 | -41.90814919465542 | 0.2414520122079263
optimization step | 88 | -42.17453527116203 | 0.10442872346500422
optimization step | 89 | -42.37581840983037 | 0.30307764523697317
optimization step | 90 | -41.91391167446115 | 0.1755909570364633
optimization step | 91 | -41.69197773337249 | 0.2606379072472764
optimization step | 92 | -42.242217989776165 | 0.32308717251373853
optimization step | 93 | -41.945044818319275 | 0.3347559478796041
optimization step | 94 | -42.62626018273897 | 0.2618955816104731
optimization step | 95 | -42.44515294825009 | 0.25456682143614934
optimization step | 96 | -41.78353721133242 | 0.20884988017960676
optimization step | 97 | -41.81511689013606 | 0.4419078491246018
optimization step | 98 | -41.95580909729595 | 0.18505282823553013
optimization step | 99 | -42.468450047317326 | 0.20091598552239562
optimization step | 100 | -42.02851770458858 | 0.29907386374740563
optimization step | 101 | -41.92540854716583 | 0.3252002405312102
optimization step | 102 | -41.91561582697385 | 0.2557943918543423
optimization step | 103 | -42.09806383622992 | 0.17223330153175806
optimization step | 104 | -41.81470836815906 | 0.2899457513919553
optimization step | 105 | -42.42889652841083 | 0.1851306980425102
optimization step | 106 | -41.992662319078505 | 0.3029535337632374
optimization step | 107 | -42.20181532566946 | 0.2376367097010879
optimization step | 108 | -42.10931864373154 | 0.1962085815020253
optimization step | 109 | -42.003855253891004 | 0.3318497089121042
optimization step | 110 | -42.420288836018436 | 0.24217556286684894
optimization step | 111 | -42.60048726177919 | 0.20818222214860768
optimization step | 112 | -42.19904015758591 | 0.22237382317486384
optimization step | 113 | -42.07956196882969 | 0.2187200732154492
optimization step | 114 | -42.00340985674226 | 0.09175827154334538
optimization step | 115 | -41.92994718423889 | 0.14705074334627938
optimization step | 116 | -42.46010930534641 | 0.2236991039206618
optimization step | 117 | -42.26370512594879 | 0.3753431614169764
optimization step | 118 | -42.11075949723378 | 0.1866765794218605
optimization step | 119 | -42.260383471267374 | 0.31542935364395414
optimization step | 120 | -42.13923074893628 | 0.3842986157034109
optimization step | 121 | -42.03060318136309 | 0.2440993519209475
optimization step | 122 | -42.365348389099864 | 0.1616649762285777
optimization step | 123 | -42.47734694770532 | 0.24742923534220534
optimization step | 124 | -42.05755427080664 | 0.2606274744881524
optimization step | 125 | -42.52334628276259 | 0.29418113382005223
optimization step | 126 | -41.88926192523026 | 0.35914722086131357
optimization step | 127 | -42.40791513529076 | 0.29517543498297166
optimization step | 128 | -42.865508118275834 | 0.22109380282696373
optimization step | 129 | -42.687163427845114 | 0.23393038722749956
optimization step | 130 | -42.30647940863955 | 0.26609016728882406
optimization step | 131 | -42.08289495190907 | 0.22459426155341994
optimization step | 132 | -42.30449869874738 | 0.3062854631650217
optimization step | 133 | -42.10855812183569 | 0.2477350046711987
optimization step | 134 | -42.14692367619424 | 0.4669271798039302
optimization step | 135 | -41.991890630045674 | 0.40046453400485116
optimization step | 136 | -42.14851501954959 | 0.22272227943067324
optimization step | 137 | -41.745980439864965 | 0.30644964330119595
optimization step | 138 | -41.7487808769938 | 0.17499736882893568
optimization step | 139 | -42.31229582819574 | 0.13931918257470818
optimization step | 140 | -41.57970399649012 | 0.3014958257954286
optimization step | 141 | -41.65105152745713 | 0.312317489481239
optimization step | 142 | -42.29312745781337 | 0.2818069365987603
optimization step | 143 | -42.08535482886496 | 0.31631456682727627
optimization step | 144 | -42.024097962255844 | 0.20904507767416972
optimization step | 145 | -41.99328430406698 | 0.3131964346091199
optimization step | 146 | -41.646697951071374 | 0.30478699705868495
optimization step | 147 | -41.99417349250256 | 0.25893349710656344
optimization step | 148 | -42.233969274425675 | 0.14691934747503227
optimization step | 149 | -41.52579034876105 | 0.19432506528288346
optimization step | 150 | -42.09833879183389 | 0.3329590240408144
optimization step | 151 | -42.053850791234446 | 0.1137195029535724
optimization step | 152 | -41.97447431955999 | 0.2876777436075087
optimization step | 153 | -42.02235340850046 | 0.23000847219402804
optimization step | 154 | -42.164722521723974 | 0.277315848361059
optimization step | 155 | -42.3278252989896 | 0.14609524965163379
optimization step | 156 | -42.3081709199157 | 0.20311443453923964
optimization step | 157 | -42.44788639883271 | 0.3730651552876614
optimization step | 158 | -41.87121961348192 | 0.16296460628919626
optimization step | 159 | -42.05289739766805 | 0.15951844475381374
optimization step | 160 | -42.63576752293144 | 0.22077377577923654
optimization step | 161 | -41.84320980340756 | 0.22124147263878466
optimization step | 162 | -41.8096182172933 | 0.36851211086667807
optimization step | 163 | -42.197357954746515 | 0.2764027845305723
optimization step | 164 | -41.97297849667257 | 0.22511294678669183
optimization step | 165 | -41.81159269840335 | 0.20071207545290085
optimization step | 166 | -41.990707282296356 | 0.17181386043344526
optimization step | 167 | -41.885304838375504 | 0.35303103675154324
optimization step | 168 | -42.29000042878594 | 0.25849985899566935
optimization step | 169 | -42.52889575130076 | 0.39512743893332886
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/li_6_exp/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 3,
    "n_proton": 3,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "n_blocks": 32,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 16,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 50,
        "print_local_energy": true,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.0,
        "include_3body": false,
        "model_string": "o"
    },
    "wave_function": {
        "add_partition_jastro": true,
        "coefficients": [
            0.3333,
            0.3333,
            0.3333
        ],
        "jastro_list": [
            "2b",
            "3b",
            "sigma",
            "tau",
            "sigma_tau"
        ],
        "n_dense": 4,
        "n_hidden_layers": 1,
        "orbitals": [
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y11_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y11_d_p"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y10_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y10_d_p"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R1_Y1m1_d_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y1m1_d_p"
            ]
        ],
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/li_6_exp/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -42.10549788239698 | 0.11201633988034565
optimization step | 1 | -41.97562009094744 | 0.1358737379529115
optimization step | 2 | -41.78697392836112 | 0.11267463981473645
optimization step | 3 | -42.065240599050526 | 0.14161946894205202
optimization step | 4 | -42.1522531848652 | 0.1483612238005919
optimization step | 5 | -42.30078910294349 | 0.1351437965198256
optimization step | 6 | -42.0614415030351 | 0.12889051443512634
optimization step | 7 | -42.04527169917159 | 0.12044340649396928
optimization step | 8 | -41.86924360956333 | 0.13426517262788007
optimization step | 9 | -42.252121414790636 | 0.1399485360182054
optimization step | 10 | -42.094312523524565 | 0.11265577181209414
optimization step | 11 | -42.286481324905566 | 0.16897010539778992
optimization step | 12 | -42.26682049729056 | 0.14706505958569793
optimization step | 13 | -42.16889823952846 | 0.15639381438185535
optimization step | 14 | -42.314162911525266 | 0.1301851288479085
optimization step | 15 | -42.245825193072925 | 0.1444157014554408
optimization step | 16 | -42.13323428633331 | 0.13303026137263846
optimization step | 17 | -42.15506768131957 | 0.11905193019498898
optimization step | 18 | -42.15634177837096 | 0.13869589200793989
optimization step | 19 | -41.73190133758724 | 0.13204121156133913
optimization step | 20 | -42.20132334037092 | 0.10592942671858004
optimization step | 21 | -42.167036509147096 | 0.15536665683051723
optimization step | 22 | -42.26327507818324 | 0.13860361906932744
optimization step | 23 | -42.14031009912043 | 0.10121092255972294
optimization step | 24 | -42.09112647732471 | 0.1429426455085275
optimization step | 25 | -42.12331945159134 | 0.11626153606744609
optimization step | 26 | -41.97936419727605 | 0.14216585873754775
optimization step | 27 | -42.017972373949355 | 0.1275906181302957
optimization step | 28 | -42.35312944809238 | 0.13161730487120532
optimization step | 29 | -42.39936287039322 | 0.11646805741749466
optimization step | 30 | -42.159213672368004 | 0.13172269960408367
