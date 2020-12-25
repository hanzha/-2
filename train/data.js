const fs = require('fs') // nodejs自带的包，专门读写文件，操作文件和文件夹，文件内容都是2进制存放在内存中的
const tf = require('@tensorflow/tfjs-node')

// 图片转tensor
const getTensor = (imgPath) => {
    // 把图片读成buffer; 在 Node.js中，定义了一个 Buffer 类，该类用来创建一个专门存放二进制数据的缓存区。一个 Buffer 类似于一个整数数组
    const buffer = fs.readFileSync(imgPath)
    // 再转成tensor
    // 优化：在TensorFlow中操作tensor时，最好加上tidy，它可以帮我们把中间的tensor变量清除掉，最佳实践，防止图片多时撑爆内存。
    return tf.tidy(() => {
        // buffer转化为Uint8Array格式，然后再用tensorflow自带的decodeImage方法把Uint8Array格式的图片转化为tensor
        const imgTs = tf.node.decodeImage(new Uint8Array(buffer))
        // 上面的转化不够，还需对tensor做处理；因为预训练模型对图片的输入有要求：调整图片尺寸为224*224
        const imgTsResized = tf.image.resizeBilinear(imgTs, [224, 224])
        // 2 图片格式必须是float格式的
        // 3 归一化：把较高的数值压缩到0-1之间或者-1到1之间。 // rgb颜色数值本来是0-255，先减去255的一半（往左平移一半），区间变成0-255/2；再除以255/2，就把值区间变成-1到1。div除法操作，sub减法操作
        return imgTsResized.toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3]) // 再转化成模型需要的形状，1：代表在图片的基础上再向外拓展一维，相当于把图片放到数组里，数组的长度只有1； 3：代表rgb彩色图片，3层矩阵；
    })
}

const getData = async (trainDir, outputDir) => {
    // 读取目录下的四个类别，数组
    const classes = fs.readdirSync(trainDir).filter(n => !n.includes('.')) // 读取文件夹下所有的文件和文件夹，同步读
    // console.log('类目：', classes)
    // 拿到类别，写文件
    fs.writeFileSync(`${outputDir}/classes.json`, JSON.stringify(classes))

    const data = []
    // 读每个类别目录下的图片
    classes.forEach((dir, dirIndex) => {
        let fileList = fs.readdirSync(`${trainDir}/${dir}`)
            .filter(n => n.match(/jpg$/))
            // .slice(0, 10)
        fileList.forEach(fileName => {
            // 图片路径
            const imgPath = `${trainDir}/${dir}/${fileName}`
            data.push({ imgPath, dirIndex })
        })
    })

    // 洗牌，样本（训练集）分布足够的随机化，这样分批采样（即读取数据）时候才更有代表性
    tf.util.shuffle(data)

    // 图片一直读会把内存撑爆，所以采取分批读取：创建一个tf的数据集，利用生成器函数的特性分批读取图片。
    const batchSize = 32 // 分批读取图片的数量
    const ds = tf.data.generator(function* () { // 利用 tf.data.generator 创建 dataset
        const count = data.length
        for (let start = 0; start < count; start += batchSize) {
            const end = Math.min(start + batchSize, count)
            console.log('当前批次：', start)
            // yield 设置每次返回的tensor； 涉及到tensor操作，放到tidy里
            yield tf.tidy(() => {
                const inputs = []
                const labels = []
                for (let j = start; j < end; j += 1) {
                    const { imgPath, dirIndex } = data[j]
                    const x = getTensor(imgPath)
                    inputs.push(x)
                    labels.push(dirIndex)
                }
                // 转成更高维的tensor
                const xs = tf.concat(inputs)
                const ys = tf.tensor(labels)
                return { xs, ys }
            })
        }
    })

    // 返回模型训练所需要的数据：
    return {
        ds,
        classes,
    }
}

module.exports = getData