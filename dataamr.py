from stog.data.dataset_readers.amr_parsing.io import AMRIO
from extra.utils import LongTensor
from extra.settings import PAD_IDX, PAD, OOV, OOV_IDX, BOS, BOS_IDX, \
    EOS, EOS_IDX
from tqdm import tqdm
import logging

logger = logging.getLogger(__file__)


def batch_data(amr_data, batch_size=20):
    data_train = []
    src_batch = []
    trg_batch = []
    src_batch_len = 0
    trg_batch_len = 0
    for src, trg in zip(amr_data.X_train_ints, amr_data.Y_train_ints):
        if len(src) > src_batch_len:
            src_batch_len = len(src)
        if len(trg) > trg_batch_len:
            trg_batch_len = len(trg)
        src_batch.append(src)
        trg_batch.append(trg)
        if len(src_batch) == batch_size:
            for seq in src_batch:
                seq.extend([PAD_IDX] * (src_batch_len - len(seq)))
            for seq in trg_batch:
                seq.extend([PAD_IDX] * (trg_batch_len - len(seq)))
            data_train.append((LongTensor(src_batch), LongTensor(trg_batch)))
            src_batch = []
            trg_batch = []
            src_batch_len = 0
            trg_batch_len = 0

    data_dev = []
    for src, trg in zip(amr_data.X_dev_ints, amr_data.Y_dev_ints):
        if len(src) > src_batch_len:
            src_batch_len = len(src)
        if len(trg) > trg_batch_len:
            trg_batch_len = len(trg)
        src_batch.append(src)
        trg_batch.append(trg)
        if len(src_batch) == batch_size:
            for seq in src_batch:
                seq.extend([PAD_IDX] * (src_batch_len - len(seq)))
            for seq in trg_batch:
                seq.extend([PAD_IDX] * (trg_batch_len - len(seq)))
            data_dev.append((LongTensor(src_batch), LongTensor(trg_batch)))
            src_batch = []
            trg_batch = []
            src_batch_len = 0
            trg_batch_len = 0

    data_test = []
    for src, trg in zip(amr_data.X_test_ints, amr_data.Y_test_ints):
        if len(src) > src_batch_len:
            src_batch_len = len(src)
        if len(trg) > trg_batch_len:
            trg_batch_len = len(trg)
        src_batch.append(src)
        trg_batch.append(trg)
        if len(src_batch) == batch_size:
            for seq in src_batch:
                seq.extend([PAD_IDX] * (src_batch_len - len(seq)))
            for seq in trg_batch:
                seq.extend([PAD_IDX] * (trg_batch_len - len(seq)))
            data_test.append((LongTensor(src_batch), LongTensor(trg_batch)))
            src_batch = []
            trg_batch = []
            src_batch_len = 0
            trg_batch_len = 0

    print("Training data size: %d" % (len(data_train) * batch_size))
    print("Training batch size: %d" % batch_size)
    print("Dev data size: %d" % (len(data_dev) * batch_size))
    print("Dev batch size: %d" % batch_size)
    print("Test data size: %d" % (len(data_test) * batch_size))
    print("Test batch size: %d" % batch_size)

    return data_train, data_dev, data_test


class AMRData():
    def __init__(self, train_file, dev_file, test_file, silver,
                 input_format="raw", use_silver_data=False, small=False):

        # Include atributes of each node to the linearized version of the graph
        self.use_silver_data = use_silver_data
        self.input_format = input_format
        self.small = small

        self.train_file = train_file
        self.X_train = list()
        self.Y_train = list()
        self.Y_train_tok = list()
        self.X_train_simple = list()
        self.X_train_simple_attributes = list()
        self.X_train_simple_only_nodes = list()
        self.X_train_concepts = list()
        self.X_train_ints = list()
        self.X_train_raw = list()
        self.Y_train_ints = list()
        self.amr_train = None

        self.silver_train_file = silver
        self.X_silver_train = list()
        self.Y_silver_train = list()
        self.Y_silver_train_tok = list()
        self.X_silver_train_simple = list()
        self.X_silver_train_simple_attributes = list()
        self.X_silver_train_simple_only_nodes = list()
        self.X_silver_train_concepts = list()
        self.X_silver_train_ints = list()
        self.X_silver_train_raw = list()
        self.Y_silver_train_ints = list()
        self.amr_silver_train = None

        self.dev_file = dev_file
        self.X_dev = list()
        self.Y_dev = list()
        self.Y_dev_tok = list()
        self.X_dev_simple = list()
        self.X_dev_simple_attributes = list()
        self.X_dev_simple_only_nodes = list()
        self.X_dev_concepts = list()
        self.X_dev_ints = list()
        self.X_dev_raw = list()
        self.Y_dev_ints = list()

        self.test_file = test_file
        self.X_test = list()
        self.Y_test = list()
        self.Y_test_tok = list()
        self.X_test_simple = list()
        self.X_test_simple_attributes = list()
        self.X_test_simple_only_nodes = list()
        self.X_test_ints = list()
        self.X_test_raw = list()
        self.Y_test_ints = list()

        self.edges = list()
        self.edges_w_attributes = list()

        self.lin_to_int = {
            PAD: PAD_IDX,
            BOS: BOS_IDX,
            EOS: EOS_IDX,
            OOV: OOV_IDX}
        self.int_to_lin = {
            PAD_IDX: PAD,
            BOS_IDX: BOS,
            EOS_IDX: EOS,
            OOV_IDX: OOV}

        self.word_to_int = {
            PAD: PAD_IDX,
            BOS: BOS_IDX,
            EOS: EOS_IDX,
            OOV: OOV_IDX}
        self.int_to_word = {
            PAD_IDX: PAD,
            BOS_IDX: BOS,
            EOS_IDX: EOS,
            OOV_IDX: OOV}

    def get_list(self, amr):
        if self.input_format == "linearized_simple":
            with_attributes = False
        else:
            with_attributes = True

        dfs_list = amr.graph.get_list_node()
        out_list = list()
        for n1, t, n2 in dfs_list:
            try:
                out_list += [":"+t, n1.__repr__()]
            except BaseException:
                return None
#           If the nodes has attributes, itter through it and add it to the
#           list
            if with_attributes:
                if len(n1.attributes) > 1:
                    for attr in n1.attributes[1:]:
                        if type(attr[1]) != str():
                            attr_tmp = str(attr[1])
                        else:
                            attr_tmp = attr[1]
#                       Attach to final list
                        out_list += [":"+attr[0], attr_tmp]
        return out_list

#   Remove not needed symbols
    def simplify(self, step):
        if step.startswith(":"):
            return step, True
        step = step.replace(" ", "")
        step = step.replace('"', "")
        step = step.replace("_", " ")
        if "/" in step:
            step = step.split("/")[1]

        if step != '-':
            step = step.split("-")[0]
        return step, False

    # Main loading method
    def load_data(self):
        logger.info("Parsing and linearizing the AMR dataset")

        train_amr = AMRIO.read(self.train_file)

        for i, amr in enumerate(train_amr):
            # Raw version
            if self.small and i > 50:
                break

            raw_amr = []
            for amr_line in str(amr.graph).splitlines():
                striped_amr = amr_line.strip()
                raw_amr.append(striped_amr)
            self.X_train_raw.append(" ".join(raw_amr))

            linearized_amr = self.get_list(amr)

            self.X_train.append(linearized_amr[1:])
            self.Y_train.append(amr.sentence)
            self.Y_train_tok.append(amr.tokens)

            # Vocabulary Create dictionaries and simplify list
            simpl = list()
            simpl_only_nodes = list()
            for step in linearized_amr:
                if step not in self.lin_to_int.keys():
                    self.lin_to_int[step] = len(self.lin_to_int)
                    self.int_to_lin[len(self.int_to_lin)] = step
                # simplyfied AMR version
                step, edge = self.simplify(step)
                simpl.append(step)
                if not step.startswith(":"):
                    simpl_only_nodes.append(step)
                # Identify edges and save them
                if edge and step not in self.edges:
                    self.edges.append(step)

            self.X_train_simple.append(simpl)
            self.X_train_simple_only_nodes.append(simpl_only_nodes)

            sent = amr.sentence.split()
            for word in sent:
                if word not in self.word_to_int.keys():
                    self.word_to_int[word] = len(self.word_to_int)
                    self.int_to_word[len(self.int_to_word)] = word

        if self.use_silver_data:
            print("Processing silver data from", self.silver_train_file)
            ii = 0

            silver_train_amr = AMRIO.read(self.silver_train_file)
            for i, amr in enumerate(silver_train_amr):
                if self.small and i > 50:
                    break

                # Raw version
                raw_amr = []
                ii += 1
                linearized_amr = self.get_list(amr)
                if linearized_amr is None:
                    continue

                for amr_line in str(amr.graph).splitlines():
                    striped_amr = amr_line.strip()
                    raw_amr.append(striped_amr)
                self.X_silver_train_raw.append(" ".join(raw_amr))

                self.X_silver_train.append(linearized_amr[1:])
                self.Y_silver_train.append(amr.sentence)
                self.Y_silver_train_tok.append(amr.tokens)

                # Vocabulary Create dictionaries and simplify list
                simpl = list()
                simpl_only_nodes = list()
                for step in linearized_amr:
                    if step not in self.lin_to_int.keys():
                        self.lin_to_int[step] = len(self.lin_to_int)
                        self.int_to_lin[len(self.int_to_lin)] = step
                    # simplyfied AMR version
                    step, edge = self.simplify(step)
                    simpl.append(step)
                    if not step.startswith(":"):
                        simpl_only_nodes.append(step)
                    # Identify edges and save them
                    if edge and step not in self.edges:
                        self.edges.append(step)

                self.X_silver_train_simple.append(simpl)
                self.X_silver_train_simple_only_nodes.append(simpl_only_nodes)

                sent = amr.sentence.split()
                for word in sent:
                    if word not in self.word_to_int.keys():
                        self.word_to_int[word] = len(self.word_to_int)
                        self.int_to_word[len(self.int_to_word)] = word
            print("Silver data with size:", len(self.X_silver_train_raw))
        else:
            print("No silver data performed")

        dev_amr = AMRIO.read(self.dev_file)
        for i, amr in enumerate(dev_amr):
            if self.small and i > 50:
                break

            # Raw input
            raw_amr = []
            for amr_line in str(amr.graph).splitlines():
                striped_amr = amr_line.strip()
                raw_amr.append(striped_amr)
            self.X_dev_raw.append(" ".join(raw_amr))

            linearized_amr = self.get_list(amr)
            self.X_dev.append(linearized_amr[1:])
            self.Y_dev.append(amr.sentence)
            self.Y_dev_tok.append(amr.tokens)

            # simplyfied AMR version
            simpl = list()
            simpl_only_nodes = list()
            for step in linearized_amr:
                step, edge = self.simplify(step)
                simpl.append(step)
                if not step.startswith(":"):
                    simpl_only_nodes.append(step)
                if edge and step not in self.edges:
                    self.edges.append(step)
            self.X_dev_simple.append(simpl)
            self.X_dev_simple_only_nodes.append(simpl_only_nodes)

        test_amr = AMRIO.read(self.test_file)
        self.amr_test = test_amr
        for i, amr in enumerate(test_amr):
            if self.small and i > 50:
                break

            # Raw version
            raw_amr = []
            for amr_line in str(amr.graph).splitlines():
                striped_amr = amr_line.strip()
                raw_amr.append(striped_amr)
            self.X_test_raw.append(" ".join(raw_amr))

            linearized_amr = self.get_list(amr)
            self.X_test.append(linearized_amr[1:])
            self.Y_test.append(amr.sentence)
            self.Y_test_tok.append(amr.tokens)

            # simplyfied AMR version
            simpl = list()
            simpl_only_nodes = list()
            for step in linearized_amr:

                step, edge = self.simplify(step)
                simpl.append(step)
                if not step.startswith(":"):
                    simpl_only_nodes.append(step)

                if edge and step not in self.edges:
                    self.edges.append(step)
            self.X_test_simple.append(simpl)
            self.X_test_simple_only_nodes.append(simpl_only_nodes)

    def output_data(self, output_src_file, output_trg_file):
        print("Write linearized AMRs to file")
        F_train_src = open(output_src_file+".train", "w")
        F_train_raw_src = open(output_src_file+".amr.train", "w")
        F_train_trg = open(output_trg_file+".train", "w")
        F_train_tok_trg = open(output_trg_file+".tok.train", "w")
        F_dev_src = open(output_src_file+".dev", "w")
        F_dev_raw_src = open(output_src_file+".amr.dev", "w")
        F_dev_trg = open(output_trg_file+".dev", "w")
        F_dev_tok_trg = open(output_trg_file+".tok.dev", "w")
        F_test_src = open(output_src_file+".test", "w")
        F_test_raw_src = open(output_src_file+".amr.test", "w")
        F_test_trg = open(output_trg_file+".test", "w")
        F_test_tok_trg = open(output_trg_file+".tok.test", "w")

        print(
            "TRAIN: src lin:", len(
                self.X_train), "src amr", len(
                self.X_train_raw), "trg text", len(
                self.Y_train_tok), "trg tok", len(
                    self.Y_train_tok))
        for x, x_raw, y, y_tok in zip(
               self.X_train, self.X_train_raw, self.Y_train, self.Y_train_tok):
            print(" ".join(x), file=F_train_src)
            print(y_tok, file=F_train_trg)
            print(x_raw, file=F_train_raw_src)
            print(y_tok, file=F_train_tok_trg)

        print(
            "dev: src lin:", len(
                self.X_dev), "src amr", len(
                self.X_dev_raw), "trg text", len(
                self.Y_dev), "trg tok", len(
                    self.Y_dev_tok))
        for x, x_raw, y, y_tok in zip(
                self.X_dev, self.X_dev_raw, self.Y_dev, self.Y_dev_tok):
            print(" ".join(x), file=F_dev_src)
            print(y_tok, file=F_dev_trg)
            print(x_raw, file=F_dev_raw_src)
            print(y_tok, file=F_dev_tok_trg)

        print(
            "test: src lin:", len(
                self.X_test), "src amr", len(
                self.X_test_raw), "trg text", len(
                self.Y_test), "trg tok", len(
                    self.Y_test_tok))
        for x, x_raw, y, y_tok in zip(
                self.X_test, self.X_test_raw, self.Y_test, self.Y_test_tok):
            print(" ".join(x), file=F_test_src)
            print(y_tok, file=F_test_trg)
            print(x_raw, file=F_test_raw_src)
            print(y_tok, file=F_test_tok_trg)

        F_train_src.close()
        F_train_trg.close()
        F_train_raw_src.close()
        F_train_tok_trg.close()
        F_dev_src.close()
        F_dev_trg.close()
        F_dev_raw_src.close()
        F_dev_tok_trg.close()
        F_test_src.close()
        F_test_trg.close()
        F_test_raw_src.close()
        F_test_tok_trg.close()

    def to_ints(self):
        print("Transform to ints")
        pbar = tqdm(total=len(self.X_train)+len(self.X_dev)+len(self.X_test))

        for x, y in zip(self.X_train, self.Y_train):
            self.X_train_ints.append([self.lin_to_int[x_i]
                                      for x_i in x] + [EOS_IDX])
            self.Y_train_ints.append([self.word_to_int[y_i]
                                      for y_i in y.split()] + [EOS_IDX])
            pbar.update(1)

        for x, y in zip(self.X_dev, self.Y_dev):
            x_in = list()
            y_in = list()
            for x_i in x:
                if x_i not in self.lin_to_int.keys():
                    x_in.append(self.lin_to_int[OOV])
                else:
                    x_in.append(self.lin_to_int[x_i])
            for y_i in y:
                if y_i not in self.word_to_int.keys():
                    y_in.append(self.word_to_int[OOV])
                else:
                    y_in.append(self.word_to_int[y_i])

            self.Y_dev_ints.append(y_in + [EOS_IDX])
            self.X_dev_ints.append(x_in + [EOS_IDX])

            pbar.update(1)

        for x, y in zip(self.X_test, self.Y_test):
            x_in = list()
            y_in = list()
            for x_i in x:
                if x_i not in self.lin_to_int.keys():
                    x_in.append(self.lin_to_int[OOV])
                else:
                    x_in.append(self.lin_to_int[x_i])
            for y_i in y:
                if y_i not in self.word_to_int.keys():
                    y_in.append(self.word_to_int[OOV])
                else:
                    y_in.append(self.word_to_int[y_i])

            self.Y_test_ints.append(y_in + [EOS_IDX])
            self.X_test_ints.append(x_in + [EOS_IDX])

            pbar.update(1)

        pbar.close()
