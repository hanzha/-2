const tf = require('@tensorflow/tfjs-node')
const getData = require('./data');

const TRAIN_DIR = '垃圾分类/train';
const OUTPUT_DIR = 'output';
const MOBILENET_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json'

const main = async () => {
    // 一：加载数据
    const {ds, classes} = await getData(TRAIN_DIR, OUTPUT_DIR);
    // console.log(ds, classes)
    
    // 二：定义模型结构：截断模型+双层神经网络
    // 加载别人训练好的一个mobilenet预训练模型
    const mobilenet = await tf.loadLayersModel(MOBILENET_URL)
    // mobilenet.summary() // 查看模型结构
    // 拿到模型所有层，然后遍历，
    // mobilenet.layers.map((l, i) => [l.name, i])
    // console.log(mobilenet.layers.map((l, i) => [l.name, i]))
    // return
    
    // 定义我们自己的模型
    const model = tf.sequential()

    // 我们需要截断0-86层。 相当于获取饺子皮、馅儿
    for (let i = 0; i <= 86; i++) {
        const layer = mobilenet.layers[i]
        layer.trainable = false // 这一层不需要再加工了； 将该层设置为不可训练的
        model.add(layer) // 把这些层加到我们自己的模型里
    }
    // 至此，截断模型完成！

    // 相当于捏成烧麦：
    // 加我们自己的双层神经网络之前，先用flatten把高维数据摊平，方便后面做分类。不参与训练
    model.add(tf.layers.flatten())
    // 定义双层神经网络用来做分类任务：
    // 第一层是隐藏层。
    model.add(tf.layers.dense({
        units: 10, // 拟合复杂的非线性的趋势；神经元个数，任意设置的，作为超参数，需要根据实验结果来调整得到最好的值
        // 激活函数。用于对上一层的所有输入求加权和，然后生成一个输出值（通常为非线性值），并将其传递给下一层。
        activation: 'relu', // 如果不加，我们的模型就只能预测一些线性变化的数据
    }))
    // 第二层输出层：多分类里，把它分成多个结果； 输出到四个类别，每个类别都会得到自己的匹配度。
    model.add(tf.layers.dense({
        units: classes.length, // 不是超参，就是类别的个数。最后一层神经元的个数就是4，每个神经元都会得到一个概率来表示匹配度
        activation: 'softmax', // 多分类，要加上softmax激活函数
    }))
    // 至此，神经网络构建完成！


    // 三：训练模型，并把训练好的模型保存到文件
    // ① 定义损失函数和优化器，需要定量计算输出结果和预期结果的差别
    model.compile({
        loss: 'sparseCategoricalCrossentropy', // 损失函数：图片输出的类别和真实类别差距有多大。对于图片分类任务，有固定的损失函数。
        optimizer: tf.train.adam(), // 优化器：帮助模型调参，调大还是调小，能让损失越来越小
        metrics: ['acc'], // 准确度的度量，训练时会显示准确度。
    })
    // ② 使用Tensorflow.js的fit方法进行训练；fit：拟合的意思，让我们的模型参数尽可能的拟合我们的训练数据
    await model.fitDataset(ds, { epochs: 20 }) // ds：要被训练的数据集； epochs：训练轮数
    // await model.fit(xs, ys, { epochs: 2 })
    // ③ 使用TensorFlow.js的save方法保存模型到文件；下一次就不用再调了，直接用。
    await model.save(`file://${process.cwd()}/${OUTPUT_DIR}`) // 生成model.json文件，记录了模型的元信息； weights.bin二进制文件，存储了权重和参数
}

main();