from glob import glob

from dataset_builder import DatasetGenerator
from model_controller import ModelPredictController, ModelTrainController


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def predict():
    model = ModelPredictController();
    model.load()
    # model.model.print_summary()
    model.restoreFromBestCheckpoint()
    model.evaluateForTest()
    # model.model.print_summary()
    print('predicted sample-1=> ', model.predictOneImage("./sample_data/sample-1.jpg"))
    print('predicted sample-2=> ', model.predictOneImage("./sample_data/sample-2.jpg"))


def predictFrom(trainName, n=1):
    model = ModelPredictController();
    model.load()
    # model.model.print_summary()
    # model.restoreFromCheckpointName( trainName)
    model.restoreFromBestCheckpoint()
    # model.model.print_summary()
    print('------------- best checkpoint ------------------')
    model.evaluateForTest()
    model.restoreFromCheckpointName(trainName)
    print('------------- ', trainName, '------------------')
    for _ in range(0, n):
        model.evaluateForTest()
    # print('predicted sample-1=> ', model.predictOneImage("./sample_data/sample-1.jpg"))
    # print('predicted sample-2=> ', model.predictOneImage("./sample_data/sample-2.jpg"))


def train(trainName):
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.prepareDatasetForTrain("../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip")
    # model.trainUntil(0.01, 1)
    # model.save(trainName)


def train_curriculum():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-1', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_2():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-2', [
        #   '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        # , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_3():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-3', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        # , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        #  '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        # , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_4():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-4', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_mixed_try11():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-mixed-try11', [
        '../train-folder/dataset/mixed.zip'
    ], 0.01, 50, (25000, 500))

    '''
        Resultado:
        len 1 accuracy 0.8157894611358643 cir 0.102339186
        len 2 accuracy 0.8070175647735596 cir 0.1125731
        len 3 accuracy 0.8157894611358643 cir 0.10550683
        len 4 accuracy 0.8004385828971863 cir 0.11878655
    '''


def train_curriculum_8():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-8', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.1, 50, (25000, 500))


def train_curriculum_10():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-10', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.01, 30, (25000, 500))


def train_curriculum_13():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-13', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.05, 0.98, 30, (25000, 500), lens=[1, 2, 3, 4])


def train_1003_5__1006_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointName('from-1003-5')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('_1006_', [
        '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.98, 50, (0, 0), lens=[1, 2, 3, 4])


def train_curriculum_14():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-14', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 30), (0, 0), lens=[1, 2, 3, 4])


def train_from_1003_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1003/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('local-1005-1-', [
        '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
    ], 0.1, 0.98, (2, 50), (0.8, 0.8), lens=[1, 2, 3, 4])


def train_from_1005_1_():  # result: 0.86	0.81	0.83	0.79
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointName('local-1005-1-')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('local-1006-1-', [
        '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.1, 0.98, (2, 50), (0.8, 0.8), lens=[1, 2, 3, 4])


def level7_try1_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1006/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('level7_try1_', [
        '../train-folder/dataset/mixed.zip'
    ], 0.025, 0.99, (2, 50), (1, 1), lens=[1, 2, 3, 4])


def evaluate_best():
    model = ModelPredictController()
    model.load()
    model.restoreFromBestCheckpoint()
    model.evaluateForTest()
    model.evaluateForTest('irt_hebraica_jan2020')


def train_curriculum_21():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-21', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        # , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 20), (1, 1), lens=[1, 2, 3, 4])


def best_add_hebraica():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1006/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('nivel-6', [
        '../train-folder/tmp/nivel-6--hebraica-metade-1'
    ], 0.025, 0.99, (1, 50), (1, 1), lens=[1, 2, 3, 4])


def train_handwritten_only():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('handwritten-only', [
        '../train-folder/dataset/handwritten-only.zip'
    ], 0.1, 0.95, (2, 50), (1, 1), lens=[1, 2, 3, 4])


def train_handwritten_only_deumavez():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('ca', [
        '../train-folder/dataset/handwritten-only.zip'
    ], 0.1, 0.95, (2, 50), (1, 1), lens=[4])


def train_handwritten_only_deumavez_teach_forcing():
    model = ModelTrainController(NO_TEACH=False);
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('handwritten-only-deumavez--teach-force-final', [
        '../train-folder/dataset/handwritten-only.zip'
    ], 0.1, 0.95, (2, 50), (1, 1), lens=[1, 2, 3, 4])


def train_curriculum_21_teacher_force():
    model = ModelTrainController(NO_TEACH=False);
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-22--teacher-force', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 20), (1, 1), lens=[1, 2, 3, 4])


def train_all_combinations():
    dataset_handwritten = ['../train-folder/dataset/handwritten-only.zip']

    dataset_niveis = [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ]

    for lens in [[4], [1, 2, 3, 4]]:
        for no_teach in [True, False]:
            for niveis in [dataset_handwritten, dataset_niveis]:
                train_name = "train_200210902_{}__{}__{}".format(
                    "FIX_LEN" if len(lens) <= 1 else "INCR_LEN",
                    "NO_TEACH" if no_teach else "TEACH",
                    "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

                print("========================================================================")
                print("INICIANDO TESTE => ", train_name)

                model = ModelTrainController(NO_TEACH=no_teach)
                model.load()
                model.initTrainSession()
                model.trainOrContinueForCurriculum(train_name,
                                                   niveis, 0.1, 0.95,
                                                   (1, 50),
                                                   (25000, 5000),
                                                   lens=lens)  # testa 2 epoch por enquanto..
                #
                print("VALIDANDO => ", train_name)
                # # model.evaluateForTest('test')
                model.evaluateForTest('irt_hebraica_jan2020')
                print("FINALIZADO TESTE => ", train_name)


def train_all_combinations__um_a_um():
    dataset_handwritten = ['../train-folder/dataset/handwritten-only.zip']
    dataset_niveis = [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ]

    ## Ajustar 1 a 1
    lens = [4]  # [[4], [1, 2, 3, 4]]:
    no_teach = True  # [True, False]
    niveis = dataset_niveis  # [dataset_handwritten, dataset_niveis]

    train_name = "train_200210902_1a1_{}__{}__{}".format(
        "FIX_LEN" if len(lens) <= 1 else "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    print("========================================================================")
    print("INICIANDO TESTE => ", train_name)

    model = ModelTrainController(NO_TEACH=no_teach)
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.95,
                                       (1, 50),
                                       (25000, 5000),
                                       lens=lens)  # testa 2 epoch por enquanto..
    #
    print("VALIDANDO => ", train_name)
    # # model.evaluateForTest('test')
    model.evaluateForTest('irt_hebraica_jan2020')
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_handwritten():
    lens = [2, 4, 6, 8, 10, 12, 14, 16]
    no_teach = True
    niveis = ['../train-folder/dataset/dataset-8lines-v002-somente_handwriten.zip']
    NUM_LINES = 8

    train_name = "train_20211026_final_{}lines_{}__{}__{}".format(
        NUM_LINES,
        "FIX_LEN" if len(lens) <= 1 else "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    # model.initTrainSession()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.90,
                                       (1, 50),
                                       (25000, 5000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', len=lens[-1])
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_curriculum():
#    lens = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]   # etapas 1
#    lens = [1, 2, 4, 6, 8, 10, 12, 14, 16]   # etapas 2, 3
    lens = [16]   # etapas 4, 5 DIRETO
    lens = [2, 4, 6, 8, 10, 12, 14, 16]   # etapas 4
    no_teach = True
    niveis = [
        '../train-folder/dataset/curriculum-8-linhas--etapa-1.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-2.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-3.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-4.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_20211026_curriculum_try2_{}lines_{}__{}__{}".format(
        NUM_LINES,
        "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    # model.initTrainSession()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.90,
                                       (1, 50),
                                       (10000, 1000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_curriculum_etapa5():
#    lens = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]   # etapas 1
#    lens = [1, 2, 4, 6, 8, 10, 12, 14, 16]   # etapas 2, 3
#    lens = [16]   # etapas 4, 5 DIRETO
    lens = [2, 4, 6, 8, 10, 12, 14, 15, 16]   # etapas 5
    no_teach = True
    niveis = [
        '../train-folder/dataset/curriculum-8-linhas--etapa-1.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-2.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-3.zip',
        # '../train-folder/dataset/curriculum-8-linhas--etapa-4.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_20211026_curriculum_try2_{}lines_{}__{}__{}".format(
        NUM_LINES,
        "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    # model.initTrainSession()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       # niveis, 0.1, 0.90,
                                       niveis, 0.1, 0.9,
                                       (1, 50),
                                       (20000, 1000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_curriculum_etapa5_refinamento_1():
#    lens = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]   # etapas 1
#    lens = [1, 2, 4, 6, 8, 10, 12, 14, 16]   # etapas 2, 3
#    lens = [16]   # etapas 4, 5 DIRETO
    lens = [16]   # etapas 5
    no_teach = True
    niveis = [
        # '../train-folder/dataset/curriculum-8-linhas--etapa-1.zip',
        # '../train-folder/dataset/curriculum-8-linhas--etapa-2.zip',
        # '../train-folder/dataset/curriculum-8-linhas--etapa-3.zip',
        # '../train-folder/dataset/curriculum-8-linhas--etapa-4.zip',
        '../train-folder/dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_20211026_curriculum_try2_8lines_INCR_LEN__NO_TEACH__CURRICULUM_ref1_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.restoreFromCheckpointName('train_20211026_curriculum_try2_8lines_INCR_LEN__NO_TEACH__CURRICULUM--curriculum-8-linhas--etapa-5')
    # model.initTrainSession()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       # niveis, 0.1, 0.90,
                                       niveis, 0.1, 0.99,
                                       (1, 50),
                                       (20000, 1000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


if __name__ == '__main__':
    print('PyCharm')
    # level7_try1_()

    train_8_lines_curriculum_etapa5_refinamento_1()

    # DatasetGenerator().generate()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# TODO: Para cada novo dataset, fazer trienamento para len=1,2,3,4
# target=loss=0.1, acc=0.99

# TODO:  Por que para alguns aparece a predição UNK?

#  TODO: Fazer um treinamento corrido logando em aquivo
#       por opoch => dataset, target len, target loss, loss, acurácia, valid loss, valid acurácia,
#       por dataset => valid  testes
#
# TODO: Tentar rodar para tamanho 8 com batch size menor, de 16 (atual é 32).
