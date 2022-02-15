DATASET :=
DO_CROSS_VALIDATION :=
RES_ROOT=$(shell pwd)/outputs

ifeq ($(DO_CROSS_VALIDATION),1)
	SUFFIX=cv
	DO_CROSS_VALIDATION_FLAG=-c
else
	SUFFIX=wo_cv
	DO_CROSS_VALIDATION_FLAG=
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
	./run.sh -t $(DO_CROSS_VALIDATION_FLAG) -d $(DATASET) && touch $@

.PHONY: text_graph
text_graph: $(RES_ROOT)/.text_graph.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.text_graph.$(DATASET).$(SUFFIX).done:
	./run.sh -tg $(DO_CROSS_VALIDATION_FLAG) -d $(DATASET) && touch $@

.PHONY: text_num
text_num: $(RES_ROOT)/.text_num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.text_num.$(DATASET).$(SUFFIX).done:
	./run.sh -tn $(DO_CROSS_VALIDATION_FLAG) -d $(DATASET) && touch $@

.PHONY: text_graph_num
text_graph_num: $(RES_ROOT)/.text_graph_num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.text_graph_num.$(DATASET).$(SUFFIX).done:
	./run.sh -tgn $(DO_CROSS_VALIDATION_FLAG) -d $(DATASET) && touch $@

.PHONY: graph
graph: $(RES_ROOT)/.graph.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.graph.$(DATASET).$(SUFFIX).done:
	./run.sh -g $(DO_CROSS_VALIDATION_FLAG) -d $(DATASET) && touch $@

.PHONY: graph_num
graph_num: $(RES_ROOT)/.graph_num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.graph_num.$(DATASET).$(SUFFIX).done:
	./run.sh -gn $(DO_CROSS_VALIDATION_FLAG) -d $(DATASET) && touch $@

.PHONY: num
num: $(RES_ROOT)/.num.$(DATASET).$(SUFFIX).done

$(RES_ROOT)/.num.$(DATASET).$(SUFFIX).done:
	./run.sh -n $(DO_CROSS_VALIDATION_FLAG) -d $(DATASET) && touch $@
