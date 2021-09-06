# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 4,
    "n_proton": 2,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "local_energy_plot_limits": [
            [
                0,
                8
            ],
            [
                -54,
                -34
            ]
        ],
        "n_blocks": 8,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 16,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 50,
        "number_of_parallel_devices": 8,
        "plot_local_energy": false,
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
        "confining_factor": 0.05,
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
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y11_d_n",
                "R1_Y11_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y10_d_n",
                "R1_Y10_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y1m1_d_n",
                "R1_Y1m1_u_n"
            ]
        ],
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
## Wave Function Parameters
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 19.084422581707017 | 0.45869175723528094
optimization step | 1 | 17.099403099665075 | 0.8414432882461431
optimization step | 2 | 14.835037815780794 | 0.9309692158450537
optimization step | 3 | 14.037699871106577 | 0.5550167251459729
optimization step | 4 | 13.224399684490367 | 0.6487509900193713
optimization step | 5 | 12.005209486599378 | 0.4848952898305686
optimization step | 6 | 12.501337876339308 | 0.7098646942967752
optimization step | 7 | 10.84325875703288 | 0.5300742999987518
optimization step | 8 | 9.692500397820139 | 0.9226726306984081
optimization step | 9 | 9.333181003963857 | 0.390627890220851
optimization step | 10 | 8.208798622120657 | 0.5462585889653927
optimization step | 11 | 6.032573613477074 | 0.576534300894378
optimization step | 12 | 7.323103479322193 | 0.9462914193916332
optimization step | 13 | 5.860728595321305 | 0.7758545315359686
optimization step | 14 | 4.723666645449756 | 0.41468339404351123
optimization step | 15 | 4.624799153848123 | 0.5928678818663876
optimization step | 16 | 2.7057428219046695 | 0.6179392763519157
optimization step | 17 | 3.5324629425199934 | 0.509949824197241
optimization step | 18 | 3.070818161553386 | 0.6961114847914486
optimization step | 19 | 2.21081245750355 | 0.8393321650725429
optimization step | 20 | 2.404637179599019 | 0.504067758183665
optimization step | 21 | 1.610064560301018 | 0.5132457385030963
optimization step | 22 | 1.1164512840782577 | 0.516570499401115
optimization step | 23 | 1.008419614814369 | 0.6281461376663372
optimization step | 24 | 0.2518486954773239 | 0.4909731076506207
optimization step | 25 | -0.6107317553296624 | 0.5736922009208315
optimization step | 26 | -0.7269694434382354 | 0.7226755818384695
optimization step | 27 | -1.6634974528799444 | 0.5572836784932023
optimization step | 28 | -1.394234689073757 | 0.650236724766956
optimization step | 29 | -1.5962294451129295 | 0.30461874602195915
optimization step | 30 | -1.7902887046345386 | 0.47832592285559566
optimization step | 31 | -3.9953319111028396 | 0.543835246674774
optimization step | 32 | -3.866581444017022 | 0.2873186503089162
optimization step | 33 | -4.59838966943907 | 0.36703017511527075
optimization step | 34 | -3.8400224928425293 | 0.43509871871847317
optimization step | 35 | -5.539733926614107 | 0.6926529705661251
optimization step | 36 | -5.3718562951117725 | 0.3833743209250014
optimization step | 37 | -6.2801914851048934 | 0.6964022835739313
optimization step | 38 | -7.307964172542707 | 0.5581296402508529
optimization step | 39 | -8.095426606254405 | 0.40215049189161756
optimization step | 40 | -9.237512631758888 | 0.6708331188361006
optimization step | 41 | -10.395911591379178 | 0.4933760464213982
optimization step | 42 | -9.65725491669191 | 0.5029211974635062
optimization step | 43 | -10.732226898822136 | 0.936208351300018
optimization step | 44 | -10.361988181283811 | 0.5810883810160234
optimization step | 45 | -10.867875890769904 | 0.78374429993889
optimization step | 46 | -12.215226660658013 | 0.8147236138813758
optimization step | 47 | -14.98418858481806 | 1.1372594594914944
optimization step | 48 | -11.193207628837706 | 0.5899678227386527
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 4,
    "n_proton": 2,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "local_energy_plot_limits": [
            [
                0,
                8
            ],
            [
                -54,
                -34
            ]
        ],
        "n_blocks": 8,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 16,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 50,
        "number_of_parallel_devices": 8,
        "plot_local_energy": false,
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
        "confining_factor": 0.05,
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
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y11_d_n",
                "R1_Y11_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y10_d_n",
                "R1_Y10_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y1m1_d_n",
                "R1_Y1m1_u_n"
            ]
        ],
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -14.550793593097639 | 0.6629946190088571
optimization step | 1 | -14.769802237538745 | 0.555405960546416
optimization step | 2 | -14.438908445917367 | 0.5869860361121648
optimization step | 3 | -14.78940724878099 | 0.4928128576567652
optimization step | 4 | -16.91772871447198 | 0.6561860244203866
optimization step | 5 | -16.951533894111307 | 0.5397991809042008
optimization step | 6 | -16.464553356589523 | 0.48947785221660567
optimization step | 7 | -18.513286216186593 | 0.5418087336418069
optimization step | 8 | -19.883766585626013 | 0.6407448919453302
optimization step | 9 | -19.83100569719884 | 0.646681027594901
optimization step | 10 | -19.437966361194228 | 0.6301462334979901
optimization step | 11 | -19.96096384983857 | 0.551845952078895
optimization step | 12 | -21.83377705155782 | 0.4291111210033672
optimization step | 13 | -21.888745806165637 | 0.5736467533609507
optimization step | 14 | -22.209156283877572 | 0.5676235234066873
optimization step | 15 | -23.40454807013718 | 0.5140909414218552
optimization step | 16 | -24.50329519605479 | 0.4146734099233442
optimization step | 17 | -25.452597108930924 | 0.5084288652422757
optimization step | 18 | -25.114360156845763 | 0.8514410985052244
optimization step | 19 | -25.309435253598 | 0.2930462233077341
optimization step | 20 | -27.27280369227618 | 0.7519669110006442
optimization step | 21 | -26.152262000080405 | 0.7794514301995317
optimization step | 22 | -28.619789630503583 | 0.9683414617466444
optimization step | 23 | -28.566157770228337 | 0.7631777866479839
optimization step | 24 | -28.473917584158695 | 0.4091480779271946
optimization step | 25 | -29.41191372166446 | 0.7773413204778197
optimization step | 26 | -28.28917265102361 | 0.6221800553834163
optimization step | 27 | -29.714889980265337 | 0.706122131772069
optimization step | 28 | -29.567961826170233 | 0.563608118080069
optimization step | 29 | -29.653854349870315 | 0.31781033349657734
optimization step | 30 | -29.82487009874147 | 0.5580552021409085
optimization step | 31 | -30.910591350623243 | 0.3538311277218607
optimization step | 32 | -31.744086636952495 | 0.5779785029399529
optimization step | 33 | -31.715150027229946 | 0.5339705654571181
optimization step | 34 | -31.49792749169806 | 0.5392809267748851
optimization step | 35 | -30.87935521496918 | 0.6776508599114915
optimization step | 36 | -30.804603511929436 | 0.6116515000368522
optimization step | 37 | -32.0016254720636 | 0.39747133758686165
optimization step | 38 | -32.09168385653573 | 0.7317161985102947
optimization step | 39 | -31.889365554250272 | 0.5147580484924184
optimization step | 40 | -32.8899093826371 | 0.4910145898151858
optimization step | 41 | -31.339263707440686 | 0.7024808465360599
optimization step | 42 | -33.84447258929345 | 0.5591498105322436
optimization step | 43 | -31.703663993975987 | 0.5564919409077778
optimization step | 44 | -33.613641552302475 | 0.3176441368155977
optimization step | 45 | -32.849095553390136 | 0.5488656733946514
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 4,
    "n_proton": 2,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "local_energy_plot_limits": [
            [
                0,
                8
            ],
            [
                -54,
                -34
            ]
        ],
        "n_blocks": 8,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 16,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 50,
        "number_of_parallel_devices": 8,
        "plot_local_energy": false,
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
        "confining_factor": 0.05,
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
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y11_d_n",
                "R1_Y11_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y10_d_n",
                "R1_Y10_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y1m1_d_n",
                "R1_Y1m1_u_n"
            ]
        ],
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 4,
    "n_proton": 2,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "local_energy_plot_limits": [
            [
                0,
                8
            ],
            [
                -54,
                -34
            ]
        ],
        "n_blocks": 8,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 16,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 50,
        "number_of_parallel_devices": 8,
        "plot_local_energy": false,
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
        "confining_factor": 0.05,
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
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y11_d_n",
                "R1_Y11_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y10_d_n",
                "R1_Y10_u_n"
            ],
            [
                "R0_Y00_d_n",
                "R0_Y00_u_n",
                "R0_Y00_d_p",
                "R0_Y00_u_p",
                "R1_Y1m1_d_n",
                "R1_Y1m1_u_n"
            ]
        ],
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_6/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -32.842404259451435 | 0.3503216800177927
optimization step | 1 | -34.220104697051475 | 0.5539669999225709
optimization step | 2 | -34.14962642383958 | 0.3673426064964699
optimization step | 3 | -33.733873612280874 | 0.4406793033750196
optimization step | 4 | -34.02180845584134 | 0.4306319987737675
optimization step | 5 | -34.44046374840054 | 0.3024563763650958
optimization step | 6 | -34.131802716903735 | 0.41909099391223614
optimization step | 7 | -34.73760285074861 | 0.37034282120183776
optimization step | 8 | -34.26230961575796 | 0.41751932785107776
optimization step | 9 | -35.06851298985967 | 0.3898068517003382
optimization step | 10 | -33.67494937583227 | 0.4114103200657561
optimization step | 11 | -34.64316234741467 | 0.5912205617402504
optimization step | 12 | -34.810445672268145 | 0.34243284751454034
optimization step | 13 | -34.37897254456206 | 0.5040630312741022
optimization step | 14 | -33.420292291175485 | 0.6989159568313323
optimization step | 15 | -34.859901698417545 | 0.5801872641463838
optimization step | 16 | -34.4596534448719 | 0.3904296643989506
optimization step | 17 | -35.843073330434194 | 0.45679138001036546
optimization step | 18 | -34.76501750746575 | 0.6657043406409537
optimization step | 19 | -34.988623422269384 | 0.36259019122857694
optimization step | 20 | -35.500697190841024 | 0.21363895271711691
optimization step | 21 | -34.912645459809546 | 0.4453607428582902
optimization step | 22 | -36.16162726551699 | 0.4868552888233876
optimization step | 23 | -35.76700350283867 | 0.26042689609085956
optimization step | 24 | -35.70537435066822 | 0.2367208982050261
optimization step | 25 | -35.42613287173843 | 0.3243754214736006
optimization step | 26 | -35.429095321685864 | 0.6079974440438357
optimization step | 27 | -34.89695429118502 | 0.48828721170672257
optimization step | 28 | -35.059493515657934 | 0.21013783745751885
optimization step | 29 | -36.19652310728157 | 0.4421528603269143
optimization step | 30 | -35.36422935868022 | 0.4578583019520097
optimization step | 31 | -36.111299963918924 | 0.48047688202139927
optimization step | 32 | -35.780624018427396 | 0.26754494116770905
optimization step | 33 | -36.684895248335344 | 0.2772574815685027
optimization step | 34 | -36.209976276542825 | 0.42293011310335454
optimization step | 35 | -35.95814222130344 | 0.38336275104829054
optimization step | 36 | -36.15982235638871 | 0.3197515171202776
optimization step | 37 | -36.14277200373937 | 0.49640765176862156
optimization step | 38 | -36.34333669756016 | 0.3385418075513936
optimization step | 39 | -35.368072920998586 | 0.37057547444169775
optimization step | 40 | -35.2854336332397 | 0.3297809106866154
optimization step | 41 | -36.31945315197867 | 0.31396523752005473
optimization step | 42 | -36.290929802676054 | 0.5056266151403289
optimization step | 43 | -35.849194357824686 | 0.3200626988737943
optimization step | 44 | -35.88394060942804 | 0.4442668260020971
optimization step | 45 | -36.04214293627748 | 0.48561924003065143
optimization step | 46 | -35.95194235927892 | 0.2392288855929625
optimization step | 47 | -36.42342865330446 | 0.34677260300720464
optimization step | 48 | -36.569325088634386 | 0.4689584408654704
optimization step | 49 | -36.0181959086951 | 0.39209816420954613
optimization step | 50 | -36.158225094490014 | 0.35382413612067476
optimization step | 51 | -36.59059109340956 | 0.41281505743374397
optimization step | 52 | -35.98134761191214 | 0.25488377757325653
optimization step | 53 | -36.37929379215884 | 0.4813537624976524
optimization step | 54 | -36.42565542120361 | 0.44262584118869663
optimization step | 55 | -36.05028291046498 | 0.333910047990746
optimization step | 56 | -36.524179422341234 | 0.4074716783208245
optimization step | 57 | -36.14844129580407 | 0.42000020493238555
optimization step | 58 | -35.770288240794855 | 0.27091775381566846
optimization step | 59 | -36.28796254911843 | 0.2984040337690835
optimization step | 60 | -36.04913124768754 | 0.4870444389361898
optimization step | 61 | -35.97435902547946 | 0.2708267107484732
optimization step | 62 | -36.54076387407334 | 0.2709966815199896
optimization step | 63 | -36.51126884494099 | 0.35724684794049444
optimization step | 64 | -36.833091424101056 | 0.3439408271787321
optimization step | 65 | -37.279904735680354 | 0.16936839434660309
optimization step | 66 | -36.76845267398211 | 0.3113221286681503
optimization step | 67 | -36.87318468963667 | 0.2823664853713856
optimization step | 68 | -36.980360089244435 | 0.389501477229485
optimization step | 69 | -35.868538858929206 | 0.2890659186722805
optimization step | 70 | -37.15694397203728 | 0.3068923481548526
optimization step | 71 | -37.026283205904235 | 0.3768596005748393
optimization step | 72 | -36.58556065814065 | 0.3989126352910003
optimization step | 73 | -36.113355633440456 | 0.4470806396025183
optimization step | 74 | -35.90722986732944 | 0.3623980194278046
optimization step | 75 | -35.872286503485505 | 0.3516502069067581
optimization step | 76 | -37.176712917554944 | 0.3244646996197523
optimization step | 77 | -36.28510356385045 | 0.38117982412578344
optimization step | 78 | -36.33773952918169 | 0.39616206219581834
optimization step | 79 | -36.71793167880324 | 0.25459900404697283
optimization step | 80 | -36.557670058220445 | 0.2548945361964334
optimization step | 81 | -36.156085777081785 | 0.11645519562386722
optimization step | 82 | -36.242697719628765 | 0.4262785027246434
optimization step | 83 | -36.96881242088535 | 0.21340950879274337
optimization step | 84 | -36.385460982133765 | 0.4152975965188192
optimization step | 85 | -36.97865263018849 | 0.22076627146083963
optimization step | 86 | -35.802263087471914 | 0.25211584926118275
optimization step | 87 | -36.434220510327954 | 0.3865285457005542
optimization step | 88 | -36.433732274483944 | 0.46476196754160815
optimization step | 89 | -37.138156255051705 | 0.2698943781566698
optimization step | 90 | -36.606342314929435 | 0.4421187293623108
optimization step | 91 | -36.44190556849627 | 0.23842496805994048
optimization step | 92 | -37.271766497714985 | 0.22296689704453845
optimization step | 93 | -36.98835645533306 | 0.2573713515694941
optimization step | 94 | -37.38584055486629 | 0.38277928628257707
optimization step | 95 | -36.67249242840714 | 0.2211702797703958
optimization step | 96 | -36.66873223069749 | 0.3925705155023235
optimization step | 97 | -36.190957844356795 | 0.4520086233677387
optimization step | 98 | -37.269618300945815 | 0.47335837848816237
optimization step | 99 | -36.79817952326532 | 0.2227026068433371
optimization step | 100 | -37.32659524260816 | 0.24781661625751983
optimization step | 101 | -36.93553988795041 | 0.33140048834418223
optimization step | 102 | -37.02442711710083 | 0.350386979234227
optimization step | 103 | -36.697022186180384 | 0.2807606710226867
optimization step | 104 | -37.02904206563756 | 0.21647366356329276
optimization step | 105 | -36.68698654325278 | 0.11617780092206376
optimization step | 106 | -37.064750881242304 | 0.38470928294002404
optimization step | 107 | -37.267801423692916 | 0.24955343521894563
optimization step | 108 | -37.22154016437057 | 0.4144771652845259
optimization step | 109 | -37.35855911124525 | 0.3030013937401118
optimization step | 110 | -36.82069877885679 | 0.3981923108309655
optimization step | 111 | -37.29662407041075 | 0.3300546209938196
optimization step | 112 | -37.01289516211652 | 0.25373773122208326
optimization step | 113 | -37.14531471687943 | 0.3620956610155189
optimization step | 114 | -37.0394740368435 | 0.3483758791894646
optimization step | 115 | -36.88093529697446 | 0.2851936669085613
optimization step | 116 | -37.26269205063687 | 0.3449873257487501
optimization step | 117 | -37.49400885004341 | 0.17605142404815127
optimization step | 118 | -36.19588224871317 | 0.4661828741811911
optimization step | 119 | -36.817870584370866 | 0.29209613481968366
optimization step | 120 | -36.788490758411314 | 0.31486006223358476
optimization step | 121 | -36.50393210622165 | 0.42242660420165806
optimization step | 122 | -36.52942239008762 | 0.30066253529118736
optimization step | 123 | -37.17449756403264 | 0.24897089619841595
optimization step | 124 | -36.529013922954555 | 0.22194691928659063
optimization step | 125 | -37.031148825794396 | 0.3502555883325107
optimization step | 126 | -37.311303793795 | 0.3072773746499931
optimization step | 127 | -36.353516647946485 | 0.2306488305546809
optimization step | 128 | -36.61944869388167 | 0.3629646155648612
optimization step | 129 | -36.810477865498555 | 0.19926571978394572
optimization step | 130 | -36.75102272271183 | 0.24909475920317212
optimization step | 131 | -36.324493686152245 | 0.3189125216999791
optimization step | 132 | -37.04151740103728 | 0.21752702700181967
optimization step | 133 | -36.9473182342303 | 0.335340484554232
optimization step | 134 | -36.81930320680934 | 0.17164993548255836
optimization step | 135 | -36.84479493723166 | 0.3100108025194751
optimization step | 136 | -36.981215775188545 | 0.3224091921351952
optimization step | 137 | -36.98270491205773 | 0.34337350351387014
