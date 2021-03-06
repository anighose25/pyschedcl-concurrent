{
	"src": "kernels.cl",
	"name": "myGEMM2",
	"inputBuffers": [{
		"break": 0,
		"type": "float",
		"pos": 3,
		"size": "16"
	},
  {
    "break": 0,
    "type": "float",
    "pos": 4,
    "size": "16"
  }
  ],
	"varArguments": [
	{
		"type": "int",
		"pos": 0,
		"value": "4"
	},
	{
		"type": "int",
		"pos": 1,
		"value": "4"
	},
	{
		"type": "int",
		"pos": 2,
		"value": "4"
	}
  ],
	"partition":5,
	"workDimension": 2,
	"globalWorkSize": [2,2],
	"id": 1,
  "outputBuffers": [
      {
          "break": 1,
          "pos": 5,
          "size": "16",
          "type": "float"
      }
  ],
	"macros_values":
      {
					 "gOutputSize" : 16,
					 "KERNEL" : 2,
					 "TS" : 2,
					 "WPT": 1,
					 "RTS": 2,
					 "PADDINGX" : 2,
					 "PADDINGY" :  2,
					 "TRANSPOSEY" : 2,
					 "TRANSPOSEX" : 2
				 }

}
