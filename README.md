# Station2vec: Regional function discovery based on metro passenger flow data embedding
The Station2vec framework consists of two components, for details please see the code: 
* station corpus extraction
* station embedding.<br>
### Data description
The data utilized is the 30 days' transaction records of smart card in April 2015, which is provided by Shanghai Public Transportation Card Company. For the file of each days' records (more than 800 MB) exceeds the limits, so can not upload. All the data used is more than 23 GB.<br>
<br>
The records is just as follows:<br>
卡号	        日期	       时间	      站点	  行业	  金额<br>
602141128	  2015-04-01	 07:51:08	   莘庄	   地铁	    0.0<br>
602141128	  2015-04-01	 09:07:57	  昌吉东路	地铁	   6.0<br>
202608138	  2015-04-01	 10:27:06	    无	     轮渡	   1.0<br>
2603250687	2015-04-01	 04:51:00	    无	     出租	   21.0<br>
### Run dependence
* python (2.7.13)
* pandas (0.20.3)
* numpy (1.13.3)
* scipy (0.19.1)
* gensim (3.1.0)
### Remarks
The code is for the International Conference on Data Science(ICDS 2018) accepted paper: "Station2vec: Regional function discovery based on metro passenger flow data embedding"

