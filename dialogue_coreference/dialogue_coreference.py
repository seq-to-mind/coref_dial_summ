import re
import pickle
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref


class NeuralCoreferenceProcessing:
    def __init__(self, gpu_id=-1):
        """ download and indicate the path of pre-trained coref-spanbert model """
        self.predictor = Predictor.from_path("./coref-spanbert-large-2021.03.10.tar.gz", cuda_device=gpu_id)

    def process(self, input_text_file_path):
        output_list = []
        input_list = open(input_text_file_path, encoding="utf-8").readlines()

        for tmp_i in tqdm(input_list):
            """ each line is one dialogue content """
            tmp_content = tmp_i.split(" }  # ")
            assert len(tmp_content) == 2
            tmp_content = tmp_content[1].replace("\n", " ").replace("🙂", " ")
            tmp_content = re.sub("\s+", " ", tmp_content).strip()

            """ here we replace the sentence segmenter, to obtain multiple coreference resolution outputs """
            tmp_res_with_dot = self.predictor.predict(tmp_content.replace("#", "."))
            tmp_res_with_sharp = self.predictor.predict(tmp_content)
            tmp_res_with_newline = self.predictor.predict(tmp_content.replace("#", "\n"))
            tmp_res_with_semicolon = self.predictor.predict(tmp_content.replace("#", ";"))

            """ check the length of multiple coreference resolution outputs are the same """
            assert len(tmp_res_with_dot['document']) == len(tmp_res_with_sharp['document'])
            assert len(tmp_res_with_newline['document']) == len(tmp_res_with_sharp['document'])
            assert len(tmp_res_with_semicolon['document']) == len(tmp_res_with_sharp['document'])

            tmp_res_with_dot['document'] = tmp_res_with_sharp['document']
            tmp_res_with_newline['document'] = tmp_res_with_sharp['document']
            tmp_res_with_semicolon['document'] = tmp_res_with_sharp['document']

            """ ensemble multiple coreference resolution outputs """
            output_list.append({"dot": (tmp_content, tmp_res_with_dot),
                                "sharp": (tmp_content, tmp_res_with_sharp),
                                "newline": (tmp_content, tmp_res_with_newline),
                                "semicolon": (tmp_content, tmp_res_with_semicolon)}, )

        return output_list


if __name__ == "__main__":
    """ you can run it individually to check the coreference resolution output """
    coref_model = NeuralCoreferenceProcessing(gpu_id=0)
    result = coref_model.process("./data/SAMsum_data/text_test.source")
    print(result[:5])
