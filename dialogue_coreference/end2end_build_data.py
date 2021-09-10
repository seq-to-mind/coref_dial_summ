from dialogue_coreference import NeuralCoreferenceProcessing
from reading_and_writing_as_input_keep_SPAN import BuildSampleWithCoreferenceInfo

# indicate the task type here: train/val/test
task_name = "test"
input_file_path = "data/SAMsum_data/text_" + task_name + ".source"

coref_model = NeuralCoreferenceProcessing(gpu_id=0)
coref_result = coref_model.process(input_file_path)

sample_building_module = BuildSampleWithCoreferenceInfo()
sample_building_module.build_sample_with_coref_to_file(task_name=task_name, input_list=coref_result)
