#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
using namespace std;

//важни променливи за мрежата
#define MAX_ERROR 0.1
#define LEARNING_SPEED 0.1
#define MOMENTUM 0.0
#define SIZE_OF_SET 88


vector<string> splitNextLine(istream& str)
{
    vector<string> result;
    string line;
    getline(str,line);

    stringstream lineStream(line);
    if(line!=""){
    string cell;

    while(getline(lineStream,cell, '\t'))
    {

        result.push_back(cell);
    }
    if (!lineStream && cell.empty())
    {
        //ако завършва на запетая добавяме празен елемент
        result.push_back("");
    }
    }
    return result;

}



struct Connection //връзка между два неврона
{
    double weight; //тежест на връзката
    double backPropWeight; //тук се пази делтата при backpropagation
};




class Neuron;
typedef vector<Neuron> Layer; //за по-добра четимост

class Neuron
{
public:
    //конструктор селектор и мутатор
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { _outputVal = val; }
    double getOutputVal(void) const { return _outputVal; }

    void feedForward(const Layer &prevLayer);

    // изчислява градиентите за мрежата
    void gradientsOfOutputLayers(double targetVal);
    void gradientsOfHiddenLayers(const Layer &nextLayer);

    //обновява тежестите на неврона
    void updateInputWeights(Layer &prevLayer);

private:
    //тези статични променливи подлежат на промяна в мрежата
    static double learningSpeed;   // скорост на учене
    static double momentum; // засилка(momentum) - умножава последната делта
    static double transferFunction(double x); //функция на прехода
    static double transferFunctionDerivative(double x); //производна на функцията на прехода
    static double randomWeight(void) ; //функция за задаване на началните тежести

    double sumError(const Layer &nextLayer) const; //изчислява приносът за грешката на даден неврон към невроните в следващият слой

    double _outputVal;
    vector<Connection> _outputWeights;//тежести на връзките към следващия слой
    unsigned _myIndex;
    double _gradient;
};

//задаване на стойности на гореизброените важни променливи
double Neuron::learningSpeed = LEARNING_SPEED;
double Neuron::momentum = MOMENTUM;
double Neuron::transferFunction(double x){return tanh(x);}
double Neuron::transferFunctionDerivative(double x){return 1.0 - x * x;}
double Neuron::randomWeight(){ return rand() / double(RAND_MAX); }


Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        _outputWeights.push_back(Connection());
        _outputWeights.back().weight = randomWeight();
    }

    _myIndex = myIndex;
}

//изчислява новата стойност в неврона според предния слой неврони
void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n) { //изчислява сумата от предните неврони
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n]._outputWeights[_myIndex].weight;
    }

    _outputVal = Neuron::transferFunction(sum);//прилага функцията на преход върху сумата
}


double Neuron::sumError(const Layer &nextLayer) const
{
    double sum = 0.0;

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {//изчислява сумата на грешката
        sum += _outputWeights[n].weight * nextLayer[n]._gradient;
    }

    return sum;
}

//изчислява градиентата на невроните от скритите слоеве
void Neuron::gradientsOfHiddenLayers(const Layer &nextLayer)
{
    double sum = sumError(nextLayer);
    _gradient = sum * Neuron::transferFunctionDerivative(_outputVal);
}

//изчислява градиентата на неврони в последния слой
void Neuron::gradientsOfOutputLayers(double targetVal)
{
    double delta = targetVal - _outputVal;
    _gradient = delta * Neuron::transferFunctionDerivative(_outputVal);
}

//обновява всички тежести, сочещи към определен неврон(при backpropagataion)
void Neuron::updateInputWeights(Layer &prevLayer)
{

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldbackPropWeight = neuron._outputWeights[_myIndex].backPropWeight;

        double newbackPropWeight =
                // базова част, умножена по скоростта
                learningSpeed
                * neuron.getOutputVal()
                * _gradient
                // добавяме засилката (momentum) на мрежата, умножена по старата делта
                + momentum
                * oldbackPropWeight;

        neuron._outputWeights[_myIndex].backPropWeight = newbackPropWeight;
        neuron._outputWeights[_myIndex].weight += newbackPropWeight;
    }
}







class Net
{
public:
    Net(const vector<unsigned> &topology);//конструктор
    void feedForward(const vector<double> &inputs);//"вкарва" стойности в мрежата
    void backProp(const vector<double> &targets);//променя тежестите в мрежата според искания отговор
    void getResults(vector<double> &results) const;//връща стойностите в крайния слой
    double getError(int numLines)  {   double error= _errorSum/numLines;
                                    _errorSum=0;
                                    return  error;}

private:
    vector<Layer> _layers; // вектор от вектори
    double _error;
    double _errorSum=0;
};


//double Net::_recentAverageSmoothingFactor = 100.0; // брой входни данни, върху които да се изчислява средната грешка

void Net::getResults(vector<double> &results) const
{
    results.clear();

    for (unsigned n = 0; n < _layers.back().size() - 1; ++n) {
        results.push_back(_layers.back()[n].getOutputVal());
    }
}

// topology = вектор със стойности броя на неврони във всеки слой ( напр {3,2,1} )
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {//за всеки слой
        _layers.push_back(Layer());//създаваме слоя
        unsigned numOutputs =
        layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];//ако сме в последния слой невроните имат по 0 "излизащи" връзки
                                                                //ако не сме в последния слой невроните имат "излизащи" връзки колкото са невроните в следващия слой


        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {//пълни слоя с неврони, включително и bias неврон
            _layers.back().push_back(Neuron(numOutputs, neuronNum));
            //cout << "Made a Neuron!" << endl;
        }

        // задава стойността на bias неврона да е 1.0
        _layers.back().back().setOutputVal(1.0);
    }
}

//backpropagation
void Net::backProp(const vector<double> &targets)
{

    Layer &outputLayer = _layers.back();
    _error = 0.0;

    // сумата от квадратите на грешката във всички изходни неврони
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targets[n] - outputLayer[n].getOutputVal();
        _error += delta * delta;
    }
    _error /= outputLayer.size() - 1; // взима средната стойност
    _error = sqrt(_error); // и намира корен от нея

    _errorSum+=_error;


    // Изчислява градиентите в изходния слой

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].gradientsOfOutputLayers(targets[n]);
    }

    // изчислява градиентите в скритите слоеве

    for (unsigned layerNum = _layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = _layers[layerNum];
        Layer &nextLayer = _layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].gradientsOfHiddenLayers(nextLayer);
        }
    }

    // обновява стойностите на теглата във всички слоеве без входния

    for (unsigned layerNum = _layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = _layers[layerNum];
        Layer &prevLayer = _layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

//изчислява стойностите в мрежата за даден вход
void Net::feedForward(const vector<double> &inputs)
{
    assert(inputs.size() == _layers[0].size() - 1);//проверява дали броя на стойностите е пълен

    // вкарва стойностите на първия слой от мрежата
    for (unsigned i = 0; i < inputs.size(); ++i) {
        _layers[0][i].setOutputVal(inputs[i]);
    }

    // изчислява напред по слоевете
    for (unsigned layerNum = 1; layerNum < _layers.size(); ++layerNum) {
        Layer &prevLayer = _layers[layerNum - 1];
        for (unsigned n = 0; n < _layers[layerNum].size() - 1; ++n) {
            _layers[layerNum][n].feedForward(prevLayer);
        }
    }
}


//показва данните от вектора в удобен вид
void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int countLines(string filename)
{
    int aNumOfLines = 0;
    ifstream aInputFile(filename);

    string aLineStr;
    while (getline(aInputFile, aLineStr))
        {
        if (!aLineStr.empty())
            aNumOfLines++;
        }
    aInputFile.close();
    return aNumOfLines;
}

int main()
{
    //например { 3, 2, 1 }
    vector<unsigned> topology;
    topology.push_back(8);
    topology.push_back(6);
    topology.push_back(4);
    Net myNet(topology);

    vector<double> inputs, targets, results;
    int trainingPass = 0;
    double error=100;
    int trainNum=0;
    int numLines=countLines("trainingSet.txt");
    while(error>=MAX_ERROR){
        cout<<"Iteration num:"<<trainNum;
        trainNum++;
        string line;
        ifstream trainingFile ("trainingSet.txt");
        if (trainingFile.is_open())
        {
        while ( !trainingFile.eof())
        {

            const char * stringLine=line.c_str();
            ++trainingPass;
            //cout << endl <<"Training Iteration num: "<<trainNum<< ", Pass: " << trainingPass;
            inputs.clear();

            //взимаме данните и ги подаваме в мрежата
            vector<string> lineofData = splitNextLine(trainingFile);
            for(int i=0;i<lineofData.size()-4;++i)inputs.push_back(stod(lineofData[i]));
            //showVectorVals(": Inputs:", inputs);
            myNet.feedForward(inputs);

            //взимаме резултатите от мрежата и ги показваме
            myNet.getResults(results);
            //showVectorVals("Outputs:", results);

            //даваме исканите резултати на мрежата и правим backpropagation
            targets.clear();
            for(int i=8;i<lineofData.size();++i)targets.push_back(stod(lineofData[i]));
            //showVectorVals("Targets:", targets);
            myNet.backProp(targets);



    }

    error = myNet.getError(numLines);
    cout << " had average error of: "
                    << error << endl;
    trainingFile.close();
    }
    else cout << "Unable to open file";
    }
    ifstream validationFile ("validationSet.txt");
        if (validationFile.is_open())
        {
        while ( !validationFile.eof())
        {
            cout<<endl;
            //const char * stringLine=line.c_str();
            //cout << endl <<"Training Iteration num: "<<trainNum<< ", Pass: " << trainingPass;
            inputs.clear();

            //взимаме данните и ги подаваме в мрежата
            vector<string> lineofData = splitNextLine(validationFile);
            for(int i=0;i<lineofData.size()-4;++i)inputs.push_back(stod(lineofData[i]));
            showVectorVals("Inputs:", inputs);
            myNet.feedForward(inputs);

            //взимаме резултатите от мрежата и ги показваме
            myNet.getResults(results);
            showVectorVals("Outputs:", results);

            //даваме исканите резултати на мрежата и правим backpropagation
            targets.clear();
            for(int i=8;i<lineofData.size();++i)targets.push_back(stod(lineofData[i]));
            showVectorVals("Targets:", targets);
            //myNet.backProp(targets);
        }
        }


}
