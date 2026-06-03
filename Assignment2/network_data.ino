#include <TensorFlowLite.h>
#include "network_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define NUMBER_OF_INPUTS 38
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 16*1024

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::ErrorReporter* error_reporter;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

const float X_test[5][38] = {
    -0.11024922321249885, -0.12470615670462065, -0.8703117991958905, 
    -0.7362346401101137, -0.00776224074056876, -0.004918644383724874, 
    -0.08948642202040107, -0.09507567152556495, -0.8092618187059747, 
    -0.011663642603760032, -0.036651869142258646, -0.024436507262009306, 
    -0.01238515036740332, -0.02618002418454278, -0.018609896340735923, 
    -0.04122119759327531, -0.0028174939213690777, -0.0975309439715147, 
    -0.1843323315145142, -0.12029767951855844, 1.602663889932865, 
    1.6051037177847889, -0.37436223991967527, -0.37443160310530493, 
    -0.8209970921283324, -0.016929597707948767, -0.37455970440553465, 
    0.7343425609306344, -0.8730893919645107, -1.0051099793265696, 
    -0.17441734820805624, -0.4801968475158174, -0.28910340026287856, 
    1.6087590765792643, 1.6189552037455606, -0.38763462350750655, 
    -0.3763870260680415, -0.6563667617603728, -0.11024922321249885, 
    -0.12470615670462065, 1.3931828317609012, -1.8517440856201013, 
    -0.00776224074056876, -0.004918644383724874, -0.08948642202040107, 
    -0.09507567152556495, -0.8092618187059747, -0.011663642603760032, 
    -0.036651869142258646, -0.024436507262009306, -0.01238515036740332, 
    -0.02618002418454278, -0.018609896340735923, -0.04122119759327531, 
    -0.0028174939213690777, -0.0975309439715147, -0.7257784945454897, 
    -0.3681102129968653, -0.6372092679572258, -0.6319290328885425, 
    2.7464027950834815, 2.715364578035608, 0.7712831058493207, 
    -0.349683030873482, -0.37455970440553465, 0.018658737949350947, 
    0.38253039421548457, 0.13088010212069012, -0.2802816761609994, 
    -0.4478339589698615, -0.20026454391970344, -0.6395319051152512, 
    -0.6248707997445304, -0.2571528566543879, -0.21987198863196714, 
    0.6528228780141483, -0.11024922321249885, -0.12470615670462065, 
    -0.6867852074966913, 0.7511112872365361, -0.007634819825848484, 
    -0.004918644383724874, -0.08948642202040107, -0.09507567152556495, 
    1.2356940323701657, -0.011663642603760032, -0.036651869142258646, 
    -0.024436507262009306, 0.35647454609934265, -0.02618002418454278, 
    -0.018609896340735923, -0.04122119759327531, -0.0028174939213690777, 
    -0.0975309439715147, -0.7257784945454897, -0.3681102129968653, 
    -0.6372092679572258, -0.6319290328885425, -0.37436223991967527, 
    -0.37443160310530493, 0.7712831058493207, -0.349683030873482, 
    -0.37455970440553465, -1.5538296618405114, -0.12333081460523297, 
    0.6654636698605771, -0.06855302025511306, 2.173560013252568, 
    -0.02258683123335326, -0.6395319051152512, -0.6248707997445304, 
    -0.38763462350750655, -0.3763870260680415, 0.21642633142264128, 
    -0.11024922321249885, -0.12470615670462065, -0.44208308523109213, 
    0.7511112872365361, -0.007727319233192214, -0.00478460656654299, 
    -0.08948642202040107, -0.09507567152556495, 1.2356940323701657, 
    -0.011663642603760032, -0.036651869142258646, -0.024436507262009306, 
    -0.01238515036740332, -0.02618002418454278, -0.018609896340735923, 
    -0.04122119759327531, -0.0028174939213690777, -0.0975309439715147, 
    -0.6122494603615755, -0.18913449437364369, -0.6372092679572258, 
    -0.6319290328885425, -0.37436223991967527, -0.37443160310530493, 
    0.7712831058493207, -0.349683030873482, -0.37455970440553465, 
    -1.6445501464437726, 1.2587542737799418, 1.0664013456654926, 
    -0.43907816809041417, -0.3183824047860378, -0.02258683123335326, 
    -0.6395319051152512, -0.6248707997445304, 0.29739465247136637, 
    -0.2824780036063969, 0.6528228780141483, -0.11024922321249885, 
    -0.12470615670462065, 1.3931828317609012, 0.7511112872365361, 
    -0.007608756456928428, -0.00483682909271775, -0.08948642202040107, 
    -0.09507567152556495, 1.2356940323701657, -0.011663642603760032, 
    -0.036651869142258646, -0.024436507262009306, -0.01238515036740332, 
    -0.02618002418454278, -0.018609896340735923, -0.04122119759327531, 
    -0.0028174939213690777, -0.0975309439715147, -0.7257784945454897, 
    -0.3681102129968653, -0.6372092679572258, -0.6319290328885425, 
    -0.37436223991967527, -0.37443160310530493, 0.7712831058493207, 
    -0.349683030873482, -0.37455970440553465, 0.6133819147929527, 
    0.030234195215341996, -0.06958873578176746, -0.2802816761609994, 
    -0.4801968475158174, -0.28910340026287856, -0.6395319051152512, 
    -0.6248707997445304, -0.38763462350750655, -0.3763870260680415, 
    0.6528228780141483
};



const uint8_t y_test[5] = {0, 1, 1, 1, 1}; // Actual labels for each sample

void setup() {
    Serial.begin(115200);
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(network_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version does not match schema version.");
        return;
    }

    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

void loop() {
    for (uint8_t i = 0; i < 5; i++) {
        // Load the i-th test sample data into the input tensor
        for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
            input->data.f[j] = X_test[i][j];
        }

        // Run the model on this input and check for error
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Failed to invoke!");
            continue;
        }

        // Question 6: Deploying the Quantized Mode
        // (a) mplement code to obtain the prediction from the output tensor and determine the predicted class label.
        float prediction = output->data.f[1];// INSERT YOUR CODE HERE //;
        int predicted_class = (output->data.f[0] > output->data.f[1]) ? 0 : 1;// INSERT YOUR CODE HERE //;

        // (b) Implement code to output Sample #, Predicted Class, and Actual Class for each sample to the serial monitor using Serial.print function.
        // INSERT YOUR CODE HERE //
        // INSERT YOUR CODE HERE //
        // INSERT YOUR CODE HERE //
        Serial.print("Sample #");
        Serial.print(i + 1);
        Serial.print(" | Predicted Class: ");
        Serial.print(predicted_class);
        Serial.print(" | Actual Class: ");
        Serial.println(y_test[i]);


        // Delay between predictions
        delay(1000);
    }

    // Delay before repeating the tests
    delay(10000);
}
