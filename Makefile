DATASET :=
DO_CROSS_VALIDATION :=
RES_ROOT=$(shell pwd)/outputs

ifeq ($(DO_CROSS_VALIDATION),1)
	SUFFIX=cv
else
	SUFFIX=wo_cv
endif

.PHONY: all
all:
	$(MAKE) text
	$(MAKE) text_graph
	$(MAKE) text_num
	$(MAKE) text_graph_num
	$(MAKE) graph
	$(MAKE) graph_num
	$(MAKE) num

.PHONY: text
text: $(RES_ROOT)/.text.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.text.$(DATASET).$(SUFFIX).done:
	./run.sh -t -d $(DATASET) && touch $@

.PHONY: text_graph
text_graph: $(RES_ROOT)/.text_graph.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.text_graph.$(DATASET).$(SUFFIX).done:
	./run.sh -tg -d $(DATASET) && touch $@

.PHONY: text_num
text_num: $(RES_ROOT)/.text_num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.text_num.$(DATASET).$(SUFFIX).done:
	./run.sh -tn -d $(DATASET) && touch $@

.PHONY: text_graph_num
text_graph_num: $(RES_ROOT)/.text_graph_num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.text_graph_num.$(DATASET).$(SUFFIX).done:
	./run.sh -tgn -d $(DATASET) && touch $@

.PHONY: graph
graph: $(RES_ROOT)/.graph.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.graph.$(DATASET).$(SUFFIX).done:
	./run.sh -g -d $(DATASET) && touch $@

.PHONY: graph_num
graph_num: $(RES_ROOT)/.graph_num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.graph_num.$(DATASET).$(SUFFIX).done:
	./run.sh -gn -d $(DATASET) && touch $@

.PHONY: num
num: $(RES_ROOT)/.num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.num.$(DATASET).$(SUFFIX).done:
	./run.sh -n -d $(DATASET) && touch $@
