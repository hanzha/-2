import React, { PureComponent } from 'react'
import { Button, Progress } from 'antd'
import 'antd/dist/antd.css'
import * as tf from '@tensorflow/tfjs' // 引入tensorflow.js
import { file2img, getImgTensor } from './utils'
import intro from './intro'

const MODEL_URL = 'http://127.0.0.1:8082/output'
class App extends PureComponent{
    state = {}

    // 组件完全渲染好之后加载模型
    async componentDidMount() {
        // 想把模型加载到浏览器里，你需要把导出的模型文件 在某个服务器下可以通过http的方式访问。
        this.model = await tf.loadLayersModel(MODEL_URL + '/model.json')
        // this.model.summary()
        this.CLASSES = await fetch(MODEL_URL + '/classes.json').then(res => res.json())
    }

    predict = async (file) => {
        if (!file) return
        const img = await file2img(file)
        this.setState({imgSrc: img.src})

        // ui被下面的this.model.predict()耗时操作给阻断了，cpu密集型操作都可以使用setTimeOut来防止ui被阻断
        setTimeout(() => {
            // 想喂给模型预测，需要是tensor
            const pred = tf.tidy(() => {
                // 图片转tensor
                const x = getImgTensor(img)
                // 拿到预测结果。 这里在第一次上传图片时 js会被阻断
                return this.model.predict(x)
            })
            pred.print()
            // 转为数组，拿到数组第一项，遍历输出，拿到对应分类的分数
            const results = pred.arraySync()[0].map((score, i) => ({ score, label: this.CLASSES[i]})).sort((a, b) => b.score - a.score)
            console.log(results)
            this.setState({results})
        }, 0)
    }

    renderResult = (item) => {
        const score = Math.round(item.score * 100)
        return (
            <tr key={item.label}>
                <td style={{width: 80, padding: '5px 0'}}>{item.label}</td>
                <td><Progress percent={score} status={score === 100 ? 'success' : 'normal'}/></td>
            </tr>
        )
    }

    render() {
        const { imgSrc, results } = this.state
        const finalItem = results && {...results[0], ...intro[results[0].label]}
        return (
            <div style={{padding: '20px'}}>
                <Button 
                    type='primary'
                    size='large'
                    style={{width: '100%'}}
                    onClick={() => this.upload.click()}
                >点击上传图片</Button>
                <input 
                    type="file" 
                    ref={el => {this.upload = el}}
                    onChange={
                        e => this.predict(e.target.files[0])
                    }
                    style={{display: 'none'}}
                />

                {imgSrc && (
                    <img src={imgSrc} style={{maxWidth: '100%', maxHeight: 300, display: 'block', margin: '20px auto',}}></img>
                )}

                {finalItem && <div style={{ marginTop: 20 }}>
                    <h1>识别结果</h1>
                    <div style={{display: 'flex', alignItems: 'flex-start', marginTop: 20 }}>
                        <img width={120} src={finalItem.icon} alt=""/>
                        <div>
                            <h2 style={{color: finalItem.color}}>{finalItem.label}</h2>
                            <p style={{color: finalItem.color}}>{finalItem.intro}</p>
                        </div>
                    </div>
                </div>}

                {results &&  <div style={{ marginTop: 20 }}>
                    <table style={{width: '100%'}}>
                        <tbody>
                            <tr>
                                <td>类别</td>
                                <td>匹配度</td>
                            </tr>
                            {results.map(this.renderResult)}
                        </tbody>
                    </table>
                </div>}
            </div>
        )
    }
}

export default App